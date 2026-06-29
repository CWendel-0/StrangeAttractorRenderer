// Vertical pass of the two-pass separable Density Estimation filter,
// followed by brightness scaling and gamma correction.
//
// de_h_tex contains horizontally-blurred LOG-DENSITY values (already tone-mapped).
// Re-derives sigma from the original raw density so both passes use the same kernel width.
// Sigma formula matches de_h.wgsl: density-relative, 3-point neighbourhood average.

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
    blend_mode:      u32,
    max_speed_enc:   u32,
    alpha_power:     f32,
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

// AA reconstruction kernel -- see de_h.wgsl's block_density for the full
// rationale. Must match de_h.wgsl/composite_light.wgsl so the sigma estimate
// this feeds is consistent across passes.
const AA_SIGMA_SCALE: f32 = 0.5; // sigma, in supersample-texel units, relative to ss_scale

fn block_density(dpx: i32, dpy: i32) -> f32 {
    if dpx < 0 || dpy < 0 || u32(dpx) >= params.width || u32(dpy) >= params.height {
        return 0.0;
    }
    let ss = f32(params.ss_scale);
    let cx = (f32(dpx) + 0.5) * ss;
    let cy = (f32(dpy) + 0.5) * ss;
    let sigma = max(ss * AA_SIGMA_SCALE, 0.6);
    let inv2s2 = 0.5 / (sigma * sigma);
    let radius = i32(ceil(sigma * 1.5));
    let icx = i32(floor(cx));
    let icy = i32(floor(cy));

    // Accumulate as f32 to avoid u32 overflow: at 4× SS, values each up to
    // ~2 B can sum past u32 range. f32 handles up to 3.4e38 with precision
    // loss of ~1 part in 8M at these magnitudes — negligible for rendering.
    var weighted   = 0.0f;
    var weight_sum = 0.0f;
    for (var dy = -radius; dy <= radius; dy++) {
        let sy = icy + dy;
        if sy < 0 || u32(sy) >= params.ss_height { continue; }
        let ddy = f32(sy) + 0.5 - cy;
        for (var dx = -radius; dx <= radius; dx++) {
            let sx = icx + dx;
            if sx < 0 || u32(sx) >= params.ss_width { continue; }
            let ddx = f32(sx) + 0.5 - cx;
            let w = exp(-(ddx * ddx + ddy * ddy) * inv2s2);
            weighted   += w * f32(accum[(u32(sy) * params.ss_width + u32(sx)) * 2u]);
            weight_sum += w;
        }
    }
    return weighted / max(weight_sum, 1e-6) / WEIGHT_SCALE;
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

    // 3-point vertical average so speckle peaks don't suppress their own blur kernel.
    let centre = (block_density(x, y - 1) + block_density(x, y) + block_density(x, y + 1)) / 3.0;
    // Density-relative sigma: matches the horizontal pass formula.
    let density_01 = clamp(log(centre + 1.0) / max(params.log_max_density, 0.001), 0.0, 1.0);
    let sigma      = max(mix(params.max_sigma, params.min_sigma, density_01), 1e-4);
    let inv_s2 = 0.5 / (sigma * sigma);
    let radius = min(i32(ceil(sigma * 3.0)), MAX_RADIUS);

    // Vertical 1D Gaussian over the horizontally-blurred intermediate texture.
    // Channel R: blurred log-density (for color).  Channel B: blurred density³ (for alpha).
    var weighted   = 0.0;
    var weighted_a = 0.0;
    var total_w    = 0.0;
    for (var dy = -radius; dy <= radius; dy++) {
        let row = y + dy;
        var d = 0.0;
        var a = 0.0;
        if row >= 0 && u32(row) < params.height {
            let tex = textureLoad(de_h_tex, vec2<i32>(x, row), 0);
            d = tex.r;
            a = tex.b;
        }
        let w   = exp(-f32(dy * dy) * inv_s2);
        weighted   += d * w;
        weighted_a += a * w;
        total_w    += w;
    }
    let blurred_log  = weighted   / max(total_w, 1e-6);
    let brightness01 = clamp(blurred_log, 0.0, 1.0);
    let fg           = clamp(pow(brightness01, 1.0 / params.gamma) * params.brightness, 0.0, 1.0);
    let alpha = clamp(weighted_a / max(total_w, 1e-6), 0.0, 1.0);
    return vec4<f32>(vec3<f32>(fg * alpha), alpha);
}
