// GPU-side Polynomial Power discrete map.
//
// p[] flat layout (P0..P23):
//   p[0].xyzw = P0,P1,P2,P3      p[1].xyzw = P4,P5,P6,P7
//   p[2].xyzw = P8,P9,P10,P11    p[3].xyzw = P12,P13,P14,P15
//   p[4].xyzw = P16,P17,P18,P19  p[5].xyzw = P20,P21,P22,P23
//
// x' = P0  + P1·x  + P2·y  + P3·z  + P4·|x|  + P5·|y|  + P6·|z|^P7
// y' = P8  + P9·x  + P10·y + P11·z + P12·|x| + P13·|y| + P14·|z|^P15
// z' = P16 + P17·x + P18·y + P19·z + P20·|x| + P21·|y| + P22·|z|^P23

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
    let upx = u32(px); let upy = u32(py);
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

// Deterministic per-trajectory hash used to seed divergence recovery so every
// trajectory recovers to a distinct point. Without this, trajectories that
// diverge and share a small modulus would warm up to bit-identical states and
// splat duplicate points into the histogram for the rest of the render.
fn recovery_hash(seed: u32) -> f32 {
    var h = seed;
    h ^= h >> 16u; h *= 0x7feb352du;
    h ^= h >> 15u; h *= 0x846ca68bu;
    h ^= h >> 16u;
    return f32(h) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let traj = gid.x;
    if traj >= params.num_traj { return; }

    var x = states[traj].x;
    var y = states[traj].y;
    var z = states[traj].z;
    var rng = bitcast<u32>(states[traj].w);
    if rng == 0u { rng = traj + 1u; }

    let p0  = params.p[0].x; let p1  = params.p[0].y; let p2  = params.p[0].z; let p3  = params.p[0].w;
    let p4  = params.p[1].x; let p5  = params.p[1].y; let p6  = params.p[1].z; let p7  = params.p[1].w;
    let p8  = params.p[2].x; let p9  = params.p[2].y; let p10 = params.p[2].z; let p11 = params.p[2].w;
    let p12 = params.p[3].x; let p13 = params.p[3].y; let p14 = params.p[3].z; let p15 = params.p[3].w;
    let p16 = params.p[4].x; let p17 = params.p[4].y; let p18 = params.p[4].z; let p19 = params.p[4].w;
    let p20 = params.p[5].x; let p21 = params.p[5].y; let p22 = params.p[5].z; let p23 = params.p[5].w;

    for (var i = 0u; i < params.steps; i++) {
        let nx = p0  + p1*x  + p2*y  + p3*z  + p4*abs(x)  + p5*abs(y)  + p6*pow(abs(z), p7);
        let ny = p8  + p9*x  + p10*y + p11*z + p12*abs(x) + p13*abs(y) + p14*pow(abs(z), p15);
        let nz = p16 + p17*x + p18*y + p19*z + p20*abs(x) + p21*abs(y) + p22*pow(abs(z), p23);

        if abs(nx) > 1e4 || abs(ny) > 1e4 || abs(nz) > 1e4 {
            // Every coordinate is derived from a differently-salted hash of the
            // full trajectory index, so the recovered state is unique per
            // trajectory (see recovery_hash above).
            x = 0.1 + recovery_hash(traj) * 1.6;
            y = recovery_hash(traj ^ 0x9E3779B9u) * 0.48;
            z = (recovery_hash(traj ^ 0x85EBCA6Bu) - 0.5) * 0.1;
            for (var w = 0u; w < 200u; w++) {
                let wx = p0 + p1*x + p2*y + p3*z + p4*abs(x) + p5*abs(y) + p6*pow(abs(z), p7);
                let wy = p8 + p9*x + p10*y + p11*z + p12*abs(x) + p13*abs(y) + p14*pow(abs(z), p15);
                let wz = p16 + p17*x + p18*y + p19*z + p20*abs(x) + p21*abs(y) + p22*pow(abs(z), p23);
                if abs(wx) < 1e4 && abs(wy) < 1e4 && abs(wz) < 1e4 { x = wx; y = wy; z = wz; }
            }
            continue;
        }

        let speed_raw = distance(vec3<f32>(nx, ny, nz), vec3<f32>(x, y, z));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);
        atomicMax(&max_vals[1], speed_enc);
        x = nx; y = ny; z = nz;

        let clip = params.view_proj * vec4<f32>(x, y, z, 1.0);
        if clip.w <= 0.0 { continue; }
        let ndc = clip.xyz / clip.w;
        if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 { continue; }

        let fx = (ndc.x * 0.5 + 0.5) * f32(params.ss_width);
        let fy = (1.0 - (ndc.y * 0.5 + 0.5)) * f32(params.ss_height);
        if fx < 0.0 || fy < 0.0 { continue; }

        rng ^= rng << 13u; rng ^= rng >> 17u; rng ^= rng << 5u;
        let noise_m = params.p[11].w;
        let fx_n = fx + (f32(rng >> 16u) / 32768.0 - 1.0) * noise_m;
        let fy_n = fy + (f32(rng & 0xFFFFu) / 32768.0 - 1.0) * noise_m;
        let px0 = i32(fx_n); let py0 = i32(fy_n);
        let ddx = fract(fx_n); let ddy = fract(fy_n);
        let w00 = u32((1.0 - ddx) * (1.0 - ddy) * WEIGHT_SCALE);
        let w10 = u32(       ddx  * (1.0 - ddy) * WEIGHT_SCALE);
        let w01 = u32((1.0 - ddx) *        ddy  * WEIGHT_SCALE);
        let w11 = 1024u - w00 - w10 - w01;
        splat_pixel(px0,     py0,     w00, speed_enc * w00 / SPEED_SCALE);
        splat_pixel(px0 + 1, py0,     w10, speed_enc * w10 / SPEED_SCALE);
        splat_pixel(px0,     py0 + 1, w01, speed_enc * w01 / SPEED_SCALE);
        splat_pixel(px0 + 1, py0 + 1, w11, speed_enc * w11 / SPEED_SCALE);
    }

    states[traj] = vec4<f32>(x, y, z, bitcast<f32>(rng));
}
