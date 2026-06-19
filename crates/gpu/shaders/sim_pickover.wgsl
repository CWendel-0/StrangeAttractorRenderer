// GPU-side Pickover attractor (discrete map).
//
// Params in p[]:
//   p[0].x = a,  p[0].y = b,  p[0].z = c,  p[0].w = d
//
// Equations:
//   x' = sin(a·y) - z·cos(b·x)
//   y' = z·sin(c·x) - cos(d·y)
//   z' = sin(x)

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

    let a = params.p[0].x;
    let b = params.p[0].y;
    let c = params.p[0].z;
    let d = params.p[0].w;

    for (var i = 0u; i < params.steps; i++) {
        let nx = sin(a * y) - z * cos(b * x);
        let ny = z * sin(c * x) - cos(d * y);
        let nz = sin(x);

        let speed_raw = distance(vec3<f32>(nx, ny, nz), vec3<f32>(x, y, z));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);
        atomicMax(&max_vals[1], speed_enc);

        x = nx; y = ny; z = nz;

        if abs(x) > 1e6 || abs(y) > 1e6 || abs(z) > 1e6 {
            // Every coordinate is derived from a differently-salted hash of the
            // full trajectory index, so the recovered state is unique per
            // trajectory (see recovery_hash above).
            x = params.p[7].x + recovery_hash(traj) * 0.32;
            y = params.p[7].y + recovery_hash(traj ^ 0x9E3779B9u) * 0.16;
            z = params.p[7].z + (recovery_hash(traj ^ 0x85EBCA6Bu) - 0.5) * 0.02;
            for (var w = 0u; w < 200u; w++) {
                let wx = x; let wy = y; let wz = z;
                x = sin(a * wy) - wz * cos(b * wx);
                y = wz * sin(c * wx) - cos(d * wy);
                z = sin(wx);
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

        rng ^= rng << 13u; rng ^= rng >> 17u; rng ^= rng << 5u;
        let noise_m = params.p[11].w;
        let fx_n = fx + (f32(rng >> 16u) / 32768.0 - 1.0) * noise_m;
        let fy_n = fy + (f32(rng & 0xFFFFu) / 32768.0 - 1.0) * noise_m;
        let px0 = i32(fx_n);
        let py0 = i32(fy_n);
        let ddx = fract(fx_n);
        let ddy = fract(fy_n);

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

    states[traj] = vec4<f32>(x, y, z, bitcast<f32>(rng));
}
