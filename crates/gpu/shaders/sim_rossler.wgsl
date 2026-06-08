// GPU-side Rössler attractor simulation.
//
// Params in p[]:
//   p[0].x = a, p[0].y = b, p[0].z = c, p[0].w = dt
//
// Equations:
//   x' = x + dt*(-y - z)
//   y' = y + dt*(x + a*y)
//   z' = z + dt*(b + z*(x - c))

struct SimParams {
    view_proj: mat4x4<f32>,
    ss_width:  u32,
    ss_height: u32,
    steps:     u32,
    num_traj:  u32,
    p:         array<vec4<f32>, 12>,
}

@group(0) @binding(0) var<storage, read_write> states:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accum:       array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> max_vals: array<atomic<u32>, 2>;
@group(0) @binding(3) var<uniform>             params:      SimParams;

const WEIGHT_SCALE: f32 = 1024.0;
const SPEED_SCALE:  u32 = 256u;

fn splat_pixel(px: i32, py: i32, weight: u32, speed_contrib: u32) {
    if weight == 0u || px < 0 || py < 0 { return; }
    let upx = u32(px);
    let upy = u32(py);
    if upx >= params.ss_width || upy >= params.ss_height { return; }
    let base = (upy * params.ss_width + upx) * 2u;
    if atomicLoad(&accum[base]) < 0x7FFFFFFFu {
        let prev = atomicAdd(&accum[base], weight);
        atomicMax(&max_vals[0], prev + weight);
    }
    if atomicLoad(&accum[base + 1u]) < 0x7FFFFFFFu {
        atomicAdd(&accum[base + 1u], speed_contrib);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let traj = gid.x;
    if traj >= params.num_traj { return; }

    var x = states[traj].x;
    var y = states[traj].y;
    var z = states[traj].z;

    let a  = params.p[0].x;
    let b  = params.p[0].y;
    let c  = params.p[0].z;
    let dt = params.p[0].w;

    for (var i = 0u; i < params.steps; i++) {
        let vx = -y - z;
        let vy =  x + a * y;
        let vz =  b + z * (x - c);

        let nx = x + dt * vx;
        let ny = y + dt * vy;
        let nz = z + dt * vz;
        x = nx; y = ny; z = nz;

        if abs(x) > 1e6 || abs(y) > 1e6 || abs(z) > 1e6 {
            x = 0.1; y = 0.0; z = f32(traj % 256u) * 0.01;
            for (var w = 0u; w < 500u; w++) {
                let wx = x; let wy = y; let wz = z;
                x = wx + dt * (-wy - wz);
                y = wy + dt * (wx + a * wy);
                z = wz + dt * (b + wz * (wx - c));
            }
            continue;
        }

        let speed_raw = length(vec3<f32>(vx, vy, vz));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);
        atomicMax(&max_vals[1], speed_enc);

        let clip = params.view_proj * vec4<f32>(x, y, z, 1.0);
        if clip.w <= 0.0 { continue; }
        let ndc = clip.xyz / clip.w;
        if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 { continue; }

        let fx = (ndc.x * 0.5 + 0.5) * f32(params.ss_width);
        let fy = (1.0 - (ndc.y * 0.5 + 0.5)) * f32(params.ss_height);
        if fx < 0.0 || fy < 0.0 { continue; }

        let px0 = i32(fx);
        let py0 = i32(fy);
        let ddx = fract(fx);
        let ddy = fract(fy);

        let w00 = u32((1.0 - ddx) * (1.0 - ddy) * WEIGHT_SCALE);
        let w10 = u32(       ddx  * (1.0 - ddy) * WEIGHT_SCALE);
        let w01 = u32((1.0 - ddx) *        ddy  * WEIGHT_SCALE);
        let w11 = 1024u - w00 - w10 - w01;

        let sc00 = speed_enc * w00 / SPEED_SCALE;
        let sc10 = speed_enc * w10 / SPEED_SCALE;
        let sc01 = speed_enc * w01 / SPEED_SCALE;
        let sc11 = speed_enc * w11 / SPEED_SCALE;

        splat_pixel(px0,     py0,     w00, sc00);
        splat_pixel(px0 + 1, py0,     w10, sc10);
        splat_pixel(px0,     py0 + 1, w01, sc01);
        splat_pixel(px0 + 1, py0 + 1, w11, sc11);
    }

    states[traj] = vec4<f32>(x, y, z, 0.0);
}
