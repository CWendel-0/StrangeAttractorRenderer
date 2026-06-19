// GPU-side Lorenz attractor simulation.
//
// Each invocation independently integrates one Lorenz trajectory and bilinearly
// splats every step into the super-sampled accumulation histogram.  Running
// thousands of trajectories in parallel gives orders-of-magnitude more
// throughput than a single CPU thread.
//
// accum layout: interleaved u32 pairs per super-sampled pixel.
//   accum[base + 0]  density  (sum of bilinear weights, scale WEIGHT_SCALE=1024)
//   accum[base + 1]  speed    (log-encoded speed × weight / SPEED_SCALE=256)
//
// mean_speed at composite = accum[base+1] / accum[base+0]
//   = weighted_mean(speed_enc) / SPEED_SCALE  ∈ [0, ~1)
//
// SimParams.p is a vec4 array encoding attractor-specific params:
//   p[0].x = a (σ), p[0].y = b (ρ), p[0].z = c (β), p[0].w = dt

struct SimParams {
    view_proj: mat4x4<f32>,          // offset 0,  64 bytes
    ss_width:  u32,                   // offset 64
    ss_height: u32,                   // offset 68
    steps:     u32,                   // offset 72
    num_traj:  u32,                   // offset 76
    p:         array<vec4<f32>, 12>,  // offset 80, 192 bytes → total 272 bytes
}

@group(0) @binding(0) var<storage, read_write> states:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accum:       array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> max_vals: array<atomic<u32>, 2>;
@group(0) @binding(3) var<uniform>             params:      SimParams;

// Must match WEIGHT_SCALE in main.rs, de_h.wgsl, composite.wgsl, and composite_light.wgsl.
const WEIGHT_SCALE: f32 = 1024.0;

// Divisor applied to speed_enc before adding to accum[base+1].
const SPEED_SCALE: u32 = 256u;

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

    let a  = params.p[0].x;
    let b  = params.p[0].y;
    let c  = params.p[0].z;
    let dt = params.p[0].w;

    for (var i = 0u; i < params.steps; i++) {
        let vx = a * (y - x);
        let vy = b * x - y - z * x;
        let vz = x * y - c * z;

        let nx = x + dt * vx;
        let ny = y + dt * vy;
        let nz = z + dt * vz;
        x = nx; y = ny; z = nz;

        if abs(x) > 1e6 || abs(y) > 1e6 || abs(z) > 1e6 {
            // Every coordinate is derived from a differently-salted hash of the
            // full trajectory index, so the recovered state is unique per
            // trajectory (see recovery_hash above).
            x = 0.1 + (recovery_hash(traj) - 0.5) * 0.2;
            y = (recovery_hash(traj ^ 0x9E3779B9u) - 0.5) * 0.2;
            z = recovery_hash(traj ^ 0x85EBCA6Bu) * 2.56;
            for (var w = 0u; w < 500u; w++) {
                let wx = x; let wy = y; let wz = z;
                x = wx + a * dt * (wy - wx);
                y = wy + dt * (b * wx - wy - wz * wx);
                z = wz + dt * (wx * wy - c * wz);
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
