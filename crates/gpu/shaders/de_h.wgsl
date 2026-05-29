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
    _pad0:           u32,
    _pad1:           u32,
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
    return total / WEIGHT_SCALE;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) f32 {
    let px = u32(in.uv.x * f32(params.width));
    let py = u32((1.0 - in.uv.y) * f32(params.height));
    if px >= params.width || py >= params.height {
        return 0.0;
    }

    let x = i32(px);
    let y = i32(py);

    if params.log_max_density <= 0.0 { return 0.0; }

    let centre = block_density(x, y);
    let sigma  = clamp(params.max_sigma / pow(centre + 1.0, 0.25), params.min_sigma, params.max_sigma);
    let inv_s2 = 0.5 / (sigma * sigma);
    let radius = min(i32(ceil(sigma * 3.0)), MAX_RADIUS);

    // Blur log-mapped values so sparse pixels stay dark even after blurring.
    var weighted = 0.0;
    var total_w  = 0.0;
    for (var dx = -radius; dx <= radius; dx++) {
        let d     = block_density(x + dx, y);
        let log_d = log(d + 1.0) / params.log_max_density;
        let w     = exp(-f32(dx * dx) * inv_s2);
        weighted += log_d * w;
        total_w  += w;
    }
    return weighted / max(total_w, 1e-6);
}
