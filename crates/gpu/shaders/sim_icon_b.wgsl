// GPU-side Icon B attractor (Field-Golubitsky original form).
//
// Params in p[]:
//   p[0].x = degree (integer, clamped 1..16)
//   p[0].y = alpha,  p[0].z = beta,  p[0].w = lambda
//   p[1].x = gamma,  p[1].y = omega
//
// Equations:
//   r    = (x + iy)^degree
//   ‖v‖  = sqrt(x² + y²)
//   p    = lambda + alpha*‖v‖ + beta*(x*Re(r) - y*Im(r))
//   x'   = p*x + gamma*Re(r) - omega*y
//   y'   = p*y - gamma*Im(r) + omega*x
//   z'   = ‖v‖

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

    let degree = clamp(u32(params.p[0].x), 2u, 16u);
    let alpha  = params.p[0].y;
    let beta   = params.p[0].z;
    let lambda = params.p[0].w;
    let gamma  = params.p[1].x;
    let omega  = params.p[1].y;

    // Approximate equilibrium radius: lambda + alpha*R = 1  →  R = (1-lambda)/alpha
    let icon_b_rhs = (1.0 - lambda) / alpha;
    var icon_b_r_eq: f32 = 0.5;
    if abs(alpha) > 1e-6 && icon_b_rhs > 0.0 {
        icon_b_r_eq = clamp(icon_b_rhs, 0.01, 2.0);
    }

    for (var i = 0u; i < params.steps; i++) {
        var re_r = 1.0; var im_r = 0.0;
        for (var k = 0u; k < degree; k++) {
            let nr = re_r * x - im_r * y;
            let ni = re_r * y + im_r * x;
            re_r = nr; im_r = ni;
        }

        let norm_v = sqrt(x * x + y * y);
        let p      = lambda + alpha * norm_v + beta * (x * re_r - y * im_r);

        let nx = p * x + gamma * re_r - omega * y;
        let ny = p * y - gamma * im_r + omega * x;
        let nz = norm_v;

        let speed_raw = distance(vec3<f32>(nx, ny, nz), vec3<f32>(x, y, z));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);
        atomicMax(&max_vals[1], speed_enc);

        x = nx; y = ny; z = nz;

        if abs(x) > 1e4 || abs(y) > 1e4 || !((x * x + y * y) < 1e8) {
            x = icon_b_r_eq + f32(traj % 16u) * 0.003;
            y = f32(traj % 8u) * 0.002;
            z = 0.0;
            for (var w = 0u; w < 100u; w++) {
                var rr = 1.0; var ri = 0.0;
                for (var k = 0u; k < degree; k++) {
                    let nr = rr * x - ri * y;
                    let ni = rr * y + ri * x;
                    rr = nr; ri = ni;
                }
                let nv = sqrt(x * x + y * y);
                let pp = lambda + alpha * nv + beta * (x * rr - y * ri);
                let wx = pp * x + gamma * rr - omega * y;
                let wy = pp * y - gamma * ri + omega * x;
                if abs(wx) > 1e4 || abs(wy) > 1e4 || !((wx * wx + wy * wy) < 1e8) { break; }
                x = wx; y = wy; z = nv;
            }
            continue;
        }

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
