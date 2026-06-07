// Vertical pass of the two-pass separable Density Estimation filter,
// followed by brightness scaling and gamma correction.
//
// de_h_tex contains horizontally-blurred LOG-DENSITY values (already tone-mapped).
// Re-derives sigma from the original raw density so both passes use the same
// kernel width. Only gamma is applied here — log was applied in the horizontal pass.

// Must match WEIGHT_SCALE in main.rs, sim.wgsl, and de_h.wgsl.
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

// de_h_tex: the R32Float intermediate written by the horizontal pass.
@group(0) @binding(0) var de_h_tex             : texture_2d<f32>;
// accum: original supersampled histogram, used to re-derive sigma.
@group(0) @binding(1) var<storage, read> accum : array<u32>;
@group(0) @binding(2) var<uniform>       params : CompositeParams;

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
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let px = u32(in.uv.x * f32(params.width));
    let py = u32((1.0 - in.uv.y) * f32(params.height));

    if px >= params.width || py >= params.height {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if params.log_max_density <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let x = i32(px);
    let y = i32(py);

    // Sigma from original density — identical formula to the horizontal pass,
    // so both passes share the same kernel width (correct separable Gaussian).
    let centre = block_density(x, y);
    let sigma  = clamp(params.max_sigma / pow(centre + 1.0, 0.25), params.min_sigma, params.max_sigma);
    let inv_s2 = 0.5 / (sigma * sigma);
    let radius = min(i32(ceil(sigma * 3.0)), MAX_RADIUS);

    // Vertical 1D Gaussian over the horizontally-blurred intermediate texture.
    var weighted = 0.0;
    var total_w  = 0.0;
    for (var dy = -radius; dy <= radius; dy++) {
        let row = y + dy;
        var d = 0.0;
        if row >= 0 && u32(row) < params.height {
            d = textureLoad(de_h_tex, vec2<i32>(x, row), 0).r;
        }
        let w = exp(-f32(dy * dy) * inv_s2);
        weighted += d * w;
        total_w  += w;
    }
    // de_h_tex values are already log-mapped to [0,1]; brightness scales the output.
    let blurred_log = weighted / max(total_w, 1e-6);
    let v = clamp(pow(clamp(blurred_log, 0.0, 1.0), 1.0 / params.gamma) * params.brightness, 0.0, 1.0);
    return vec4<f32>(v, v, v, 1.0);
}
