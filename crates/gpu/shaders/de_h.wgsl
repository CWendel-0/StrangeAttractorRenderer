// Horizontal pass of the two-pass separable Density Estimation filter.
//
// Blurs in log-density (perceptual) space rather than raw density space.
// Dense pixels borrow very little from neighbours (narrow kernel, log scale compresses range).
// Sparse pixels are smoothed but stay dark — they can only borrow log-scaled brightness,
// so they never appear as bright as dense pixels even after blurring.
//
// Output stored in de_h_tex (R32Float) is already log-mapped; composite only applies
// the vertical blur and gamma correction.

// Must match WEIGHT_SCALE in main.rs, sim.wgsl, and composite.wgsl.
const WEIGHT_SCALE: f32 = 1024.0;
// Hard cap on kernel half-width. Supports max_sigma up to ~10 (3σ = 30).
const MAX_RADIUS:   i32 = 30;

struct CompositeParams {
    width:           u32,
    height:          u32,
    log_max_density: f32,
    brightness:      f32,
    gamma:           f32,
    ss_width:        u32,
    ss_height:       u32,
    max_sigma:       f32,
    min_sigma:       f32,
    ss_scale:        u32,
    blend_mode:      u32,
    max_speed_enc:   u32,
    alpha_power:     f32,
}

@group(0) @binding(0) var<storage, read> accum  : array<u32>;
@group(0) @binding(1) var<uniform>       params : CompositeParams;

struct VertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertOut;
    out.pos = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv  = pos[vi] * 0.5 + 0.5;
    return out;
}

fn block_density(dpx: i32, dpy: i32) -> f32 {
    if dpx < 0 || dpy < 0 || u32(dpx) >= params.width || u32(dpy) >= params.height {
        return 0.0;
    }
    let ssx = u32(dpx) * params.ss_scale;
    let ssy = u32(dpy) * params.ss_scale;
    // Accumulate as f32 to avoid u32 overflow: at 4× SS, 16 values each up to
    // ~2 B sum to ~32 B which wraps u32.  f32 handles up to 3.4e38 with
    // precision loss of ~1 part in 8M at these magnitudes — negligible for rendering.
    var total = 0.0f;
    for (var dy = 0u; dy < params.ss_scale; dy++) {
        for (var dx = 0u; dx < params.ss_scale; dx++) {
            total += f32(accum[((ssy + dy) * params.ss_width + ssx + dx) * 2u]);
        }
    }
    // Divide by ss_scale² to get per-SS-pixel average density, consistent with
    // log_max_density which tracks the max of a single SS pixel.  Without this,
    // block_density is ss_scale²× too large, inflating density_01 and squeezing
    // the entire attractor into the upper portion of the gradient range.
    let ss = f32(params.ss_scale);
    return total / WEIGHT_SCALE / (ss * ss);
}

fn block_speed(dpx: i32, dpy: i32) -> f32 {
    if dpx < 0 || dpy < 0 || u32(dpx) >= params.width || u32(dpy) >= params.height {
        return 0.0;
    }
    let ssx = u32(dpx) * params.ss_scale;
    let ssy = u32(dpy) * params.ss_scale;
    var total = 0.0f;
    for (var dy = 0u; dy < params.ss_scale; dy++) {
        for (var dx = 0u; dx < params.ss_scale; dx++) {
            total += f32(accum[((ssy + dy) * params.ss_width + ssx + dx) * 2u + 1u]);
        }
    }
    // Same ss_scale² normalisation as block_density so spd/d ratio stays correct.
    let ss = f32(params.ss_scale);
    return total / WEIGHT_SCALE / (ss * ss);
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let px = u32(in.uv.x * f32(params.width));
    let py = u32((1.0 - in.uv.y) * f32(params.height));
    if px >= params.width || py >= params.height {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let x = i32(px);
    let y = i32(py);

    if params.log_max_density <= 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }

    // 3-point horizontal average so isolated speckle peaks don't suppress their own blur kernel.
    let centre = (block_density(x - 1, y) + block_density(x, y) + block_density(x + 1, y)) / 3.0;
    // Density-relative sigma: sparse pixels get max_sigma, the densest pixel gets min_sigma.
    // Scales automatically as the render accumulates — more samples → lower sigma everywhere.
    let density_01   = clamp(log(centre + 1.0) / max(params.log_max_density, 0.001), 0.0, 1.0);
    // Floor at 1e-4 so σ=0 (no-blur mode) doesn't produce NaN via 0.5/(σ²)=∞.
    let sigma        = max(mix(params.max_sigma, params.min_sigma, density_01), 1e-4);
    // Speed channel floored at 1.5 px so fine-grain speed noise stays blurred.
    let speed_sigma  = max(sigma, 1.5);
    let inv_s2       = 0.5 / (sigma * sigma);
    let speed_inv_s2 = 0.5 / (speed_sigma * speed_sigma);
    let radius       = min(i32(ceil(sigma * 3.0)),       MAX_RADIUS);
    let speed_radius = min(i32(ceil(speed_sigma * 3.0)), MAX_RADIUS);

    let max_s = max(f32(params.max_speed_enc), 1.0);

    // Horizontal blur — density and speed use separate kernels.
    // Loop over the wider speed radius; density only accumulates within its own radius.
    // Channel B stores pow(density_01, alpha_power) blurred horizontally.  Applying the
    // power BEFORE blurring lets dense pixels spread their large alpha values outward,
    // creating a 3–6 px soft glow at the attractor boundary instead of a hard cutoff.
    // Isolated low-density hits stay invisible because their small pow(d_01, n) value
    // is further diluted by the Gaussian kernel — speckle protection comes from the
    // last_max_density floor (1024) ensuring sparse first-hit pixels have d_01 ≈ 0.1.
    var weighted_d = 0.0;
    var weighted_s = 0.0;
    var weighted_a = 0.0;
    var total_w_d  = 0.0;
    var total_w_s  = 0.0;
    for (var dx = -speed_radius; dx <= speed_radius; dx++) {
        let d      = block_density(x + dx, y);
        let spd    = block_speed(x + dx, y);
        let log_d  = log(d + 1.0) / params.log_max_density;
        let mean_s = select(0.0, spd / d * 256.0 / max_s, d > 1e-3);
        let w_s    = exp(-f32(dx * dx) * speed_inv_s2);
        weighted_s += mean_s * w_s;
        total_w_s  += w_s;
        if abs(dx) <= radius {
            let w_d   = exp(-f32(dx * dx) * inv_s2);
            weighted_d += log_d * w_d;
            total_w_d  += w_d;
            let d_01   = clamp(log_d, 0.0, 1.0);
            weighted_a += pow(d_01, params.alpha_power) * w_d;
        }
    }
    return vec4<f32>(
        weighted_d / max(total_w_d, 1e-6),
        clamp(weighted_s / max(total_w_s, 1e-6), 0.0, 1.0),
        weighted_a / max(total_w_d, 1e-6),
        0.0,
    );
}
