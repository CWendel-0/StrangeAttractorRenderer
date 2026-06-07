// GPU-side Polynomial Type C attractor (discrete map).
//
// 18 params spread across p[0..4]:
//   p[0] = {P0, P1,  P2,  P3}   X: const, lin-x, quad-x², cross-xy
//   p[1] = {P4, P5,  P6,  P7}   X: lin-y, quad-y²; Y: const, lin-y
//   p[2] = {P8, P9,  P10, P11}  Y: quad-y², cross-yz, lin-z, quad-z²
//   p[3] = {P12,P13, P14, P15}  Z: const, lin-z, quad-z², cross-zx
//   p[4] = {P16,P17, 0,   0}    Z: lin-x, quad-x²
//
// Equations:
//   x' = P0  + x*(P1  + P2*x  + P3*y)  + y*(P4  + P5*y)
//   y' = P6  + y*(P7  + P8*y  + P9*z)  + z*(P10 + P11*z)
//   z' = P12 + z*(P13 + P14*z + P15*x) + x*(P16 + P17*x)

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
@group(0) @binding(2) var<storage, read_write> max_density: atomic<u32>;
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
        atomicMax(&max_density, prev + weight);
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

    // Unpack params from vec4 array
    let p00 = params.p[0].x;  // P0
    let p01 = params.p[0].y;  // P1
    let p02 = params.p[0].z;  // P2
    let p03 = params.p[0].w;  // P3
    let p04 = params.p[1].x;  // P4
    let p05 = params.p[1].y;  // P5
    let p06 = params.p[1].z;  // P6
    let p07 = params.p[1].w;  // P7
    let p08 = params.p[2].x;  // P8
    let p09 = params.p[2].y;  // P9
    let p10 = params.p[2].z;  // P10
    let p11 = params.p[2].w;  // P11
    let p12 = params.p[3].x;  // P12
    let p13 = params.p[3].y;  // P13
    let p14 = params.p[3].z;  // P14
    let p15 = params.p[3].w;  // P15
    let p16 = params.p[4].x;  // P16
    let p17 = params.p[4].y;  // P17

    for (var i = 0u; i < params.steps; i++) {
        let nx = p00 + x * (p01 + p02 * x + p03 * y) + y * (p04 + p05 * y);
        let ny = p06 + y * (p07 + p08 * y + p09 * z) + z * (p10 + p11 * z);
        let nz = p12 + z * (p13 + p14 * z + p15 * x) + x * (p16 + p17 * x);

        let speed_raw = distance(vec3<f32>(nx, ny, nz), vec3<f32>(x, y, z));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);

        x = nx; y = ny; z = nz;

        if abs(x) > 1e6 || abs(y) > 1e6 || abs(z) > 1e6 || !(x == x) {
            x = params.p[7].x + f32(traj % 32u) * 0.01;
            y = params.p[7].y + f32(traj % 16u) * 0.01;
            z = params.p[7].z;
            for (var w = 0u; w < 200u; w++) {
                let wx = x; let wy = y; let wz = z;
                x = p00 + wx * (p01 + p02 * wx + p03 * wy) + wy * (p04 + p05 * wy);
                y = p06 + wy * (p07 + p08 * wy + p09 * wz) + wz * (p10 + p11 * wz);
                z = p12 + wz * (p13 + p14 * wz + p15 * wx) + wx * (p16 + p17 * wx);
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
