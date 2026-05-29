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

struct SimParams {
    view_proj: mat4x4<f32>,  // 64 bytes, offset 0
    ss_width:  u32,           // offset 64
    ss_height: u32,           // offset 68
    steps:     u32,
    num_traj:  u32,           // offset 76
    a:         f32,           // Lorenz σ
    b:         f32,           // Lorenz ρ
    c:         f32,           // Lorenz β
    dt:        f32,           // time-step
    // total 96 bytes (6 × 16-byte blocks) ✓
}

@group(0) @binding(0) var<storage, read_write> states:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accum:       array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> max_density: atomic<u32>;
@group(0) @binding(3) var<uniform>             params:      SimParams;

// Must match WEIGHT_SCALE in main.rs, de_h.wgsl, composite.wgsl, and composite_light.wgsl.
const WEIGHT_SCALE: f32 = 1024.0;

// Divisor applied to speed_enc before adding to accum[base+1].
// Keeps the per-splat contribution ≤ WEIGHT_SCALE so the speed slot overflows
// at the same rate as the density slot — both guarded at 0x7FFFFFFF.
const SPEED_SCALE: u32 = 256u;

fn splat_pixel(px: i32, py: i32, weight: u32, speed_contrib: u32) {
    if weight == 0u || px < 0 || py < 0 { return; }
    let upx = u32(px);
    let upy = u32(py);
    if upx >= params.ss_width || upy >= params.ss_height { return; }
    let base = (upy * params.ss_width + upx) * 2u;

    // Overflow guard on density: threshold 0x7FFFFFFF leaves 2 B headroom.
    // Even if all 8 192 threads race through the check, the worst-case overshoot
    // is 8 192 × 1 024 ≈ 8 M — far below the 2 B gap.
    if atomicLoad(&accum[base]) < 0x7FFFFFFFu {
        let prev = atomicAdd(&accum[base], weight);
        atomicMax(&max_density, prev + weight);
    }

    // Speed slot gets the same overflow guard.  With speed_contrib ≤ weight
    // (since speed_enc ≤ 255 and contribution = speed_enc * weight / SPEED_SCALE
    // ≤ 255 * 1024 / 256 = 1020 ≤ WEIGHT_SCALE), it saturates at the same rate.
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

    let a  = params.a;
    let b  = params.b;
    let c  = params.c;
    let dt = params.dt;

    for (var i = 0u; i < params.steps; i++) {
        // Compute raw derivatives (before dt scaling) — used for speed.
        let vx = a * (y - x);
        let vy = b * x - y - z * x;
        let vz = x * y - c * z;

        // Euler step.
        let nx = x + dt * vx;
        let ny = y + dt * vy;
        let nz = z + dt * vz;
        x = nx; y = ny; z = nz;

        // Divergence guard: Lorenz rarely escapes, but handle it gracefully.
        if abs(x) > 1e6 || abs(y) > 1e6 || abs(z) > 1e6 {
            x = 0.1; y = 0.0; z = f32(traj % 256u) * 0.01;
            for (var w = 0u; w < 500u; w++) {
                let wx = x; let wy = y; let wz = z;
                x = wx + a * dt * (wy - wx);
                y = wy + dt * (b * wx - wy - wz * wx);
                z = wz + dt * (wx * wy - c * wz);
            }
            // continue (not break) so remaining steps splat from the rescued position.
            continue;
        }

        // Log-encode speed so the full range of Lorenz velocities (roughly 0–1000+)
        // maps comfortably to [0, ~255].  log(speed+1)*32:
        //   speed≈10  → enc≈76,  ÷256 ≈ 0.30
        //   speed≈100 → enc≈147, ÷256 ≈ 0.57
        //   speed≈500 → enc≈198, ÷256 ≈ 0.77
        let speed_raw = length(vec3<f32>(vx, vy, vz));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);

        // Project into super-sampled screen space.
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

        // Bilinear weights summing exactly to WEIGHT_SCALE.
        // w11 absorbs the truncation remainder so the total is always 1024.
        // NOTE: atomicAdd of prev+weight is a thread-local snapshot; concurrent
        // adds to the same slot mean atomicMax may underestimate the true peak
        // by however many other threads wrote between our add and our max call.
        // The error is bounded and corrects within one readback frame.
        let w00 = u32((1.0 - ddx) * (1.0 - ddy) * WEIGHT_SCALE);
        let w10 = u32(       ddx  * (1.0 - ddy) * WEIGHT_SCALE);
        let w01 = u32((1.0 - ddx) *        ddy  * WEIGHT_SCALE);
        let w11 = 1024u - w00 - w10 - w01;

        // Speed contribution per pixel = speed_enc * bilinear_weight / SPEED_SCALE.
        // This keeps the magnitude proportional to density so both slots overflow together.
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
