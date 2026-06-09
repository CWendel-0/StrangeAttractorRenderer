// Light rendering mode: vertical DE blur + additive gradient colorization.
//
// gradient_a is sampled by normalized log-density (same as monochrome brightness).
// gradient_b is sampled by normalized mean speed (speed_fixed / density).
// The two colors are added together and gamma-corrected.
//
// Speed encoding: sim.wgsl stores  speed_enc * weight / 256  in accum[base+1],
// where speed_enc = min(u32(log(speed+1.0) * 32.0), 255u).  The ratio
// block_speed / block_density therefore lands in [0, ~1) without any extra scale.

// Must match WEIGHT_SCALE in main.rs, sim.wgsl, de_h.wgsl, and composite.wgsl.
const WEIGHT_SCALE: f32 = 1024.0;
// Hard cap on kernel half-width. Supports max_sigma up to ~10 (3σ = 30).
const MAX_RADIUS:   i32  = 30;

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
}

@group(0) @binding(0) var de_h_tex             : texture_2d<f32>;
@group(0) @binding(1) var<storage, read> accum : array<u32>;
@group(0) @binding(2) var<uniform>       params : CompositeParams;
// 256×1 Rgba8Unorm gradient textures uploaded from the CPU.
@group(0) @binding(3) var gradient_a           : texture_2d<f32>;
@group(0) @binding(4) var gradient_b           : texture_2d<f32>;
@group(0) @binding(5) var grad_sampler         : sampler;

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

// Sum the density channel (accum[base*2]) over the ss_scale² block for display pixel (dpx, dpy).
fn block_density(dpx: i32, dpy: i32) -> f32 {
    if dpx < 0 || dpy < 0 || u32(dpx) >= params.width || u32(dpy) >= params.height {
        return 0.0;
    }
    let ssx = u32(dpx) * params.ss_scale;
    let ssy = u32(dpy) * params.ss_scale;
    var total = 0.0f;
    for (var dy = 0u; dy < params.ss_scale; dy++) {
        for (var dx = 0u; dx < params.ss_scale; dx++) {
            total += f32(accum[((ssy + dy) * params.ss_width + ssx + dx) * 2u]);
        }
    }
    return total / WEIGHT_SCALE;
}

// Sample a 256×1 gradient texture at t ∈ [0, 1] using a linear sampler.
fn sample_grad(tex: texture_2d<f32>, t: f32) -> vec3<f32> {
    return textureSample(tex, grad_sampler, vec2<f32>(clamp(t, 0.0, 1.0), 0.5)).rgb;
}

// ---------------------------------------------------------------------------
// HSL blend mode helpers (W3C Compositing and Blending Level 1)
// ---------------------------------------------------------------------------

fn blm_lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn blm_clip(c: vec3<f32>) -> vec3<f32> {
    let l = blm_lum(c);
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    var r = c;
    if n < 0.0 { r = l + (r - l) * l / max(l - n, 1e-6); }
    if x > 1.0 { r = l + (r - l) * (1.0 - l) / max(x - l, 1e-6); }
    return r;
}

fn blm_set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return blm_clip(c + (l - blm_lum(c)));
}

fn blm_sat(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

fn blm_set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mn = min(c.r, min(c.g, c.b));
    let mx = max(c.r, max(c.g, c.b));
    if mx <= mn { return vec3<f32>(0.0); }
    return (c - mn) * s / (mx - mn);
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
    let centre       = (block_density(x, y - 1) + block_density(x, y) + block_density(x, y + 1)) / 3.0;
    // Density-relative sigma: matches the horizontal pass formula.
    let density_01   = clamp(log(centre + 1.0) / max(params.log_max_density, 0.001), 0.0, 1.0);
    let sigma        = mix(params.max_sigma, params.min_sigma, density_01);
    let speed_sigma  = max(sigma, 1.5);
    let inv_s2       = 0.5 / (sigma * sigma);
    let speed_inv_s2 = 0.5 / (speed_sigma * speed_sigma);
    let radius       = min(i32(ceil(sigma * 3.0)),       MAX_RADIUS);
    let speed_radius = min(i32(ceil(speed_sigma * 3.0)), MAX_RADIUS);

    // Vertical 1D Gaussian — density and speed use separate kernels (same floor logic as de_h).
    var weighted     = 0.0;
    var weighted_spd = 0.0;
    var total_w      = 0.0;
    var total_w_s    = 0.0;
    for (var dy = -speed_radius; dy <= speed_radius; dy++) {
        let row = y + dy;
        var d = 0.0;
        var s = 0.0;
        if row >= 0 && u32(row) < params.height {
            let tex = textureLoad(de_h_tex, vec2<i32>(x, row), 0);
            d = tex.r;
            s = tex.g;
        }
        let w_s = exp(-f32(dy * dy) * speed_inv_s2);
        weighted_spd += s * w_s;
        total_w_s    += w_s;
        if abs(dy) <= radius {
            let w = exp(-f32(dy * dy) * inv_s2);
            weighted += d * w;
            total_w  += w;
        }
    }
    let blurred_log   = weighted     / max(total_w, 1e-6);
    let mean_speed_01 = clamp(weighted_spd / max(total_w_s, 1e-6), 0.0, 1.0);

    // Gradient A: sample by log-density; Gradient B: sample by mean speed.
    // Both are fully DE-blurred (horizontal + vertical Gaussian).
    let grad_density_01 = clamp(blurred_log, 0.0, 1.0);
    let col_a = sample_grad(gradient_a, grad_density_01);
    let col_b = sample_grad(gradient_b, mean_speed_01);

    // Blend col_a (density) and col_b (speed) according to params.blend_mode.
    // Photoshop convention: col_a = base layer, col_b = blend layer.
    var blended: vec3<f32>;
    switch params.blend_mode {
        case 1u {  // Subtract
            blended = clamp(col_a - col_b, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        case 2u {  // Multiply
            blended = col_a * col_b;
        }
        case 3u {  // Divide
            blended = clamp(col_a / max(col_b, vec3<f32>(0.001)), vec3<f32>(0.0), vec3<f32>(1.0));
        }
        case 4u {  // Lighten
            blended = max(col_a, col_b);
        }
        case 5u {  // Darken
            blended = min(col_a, col_b);
        }
        case 6u {  // Hard Light  (col_b < 0.5 → Multiply; else → Screen)
            blended = select(
                1.0 - 2.0 * (1.0 - col_a) * (1.0 - col_b),
                2.0 * col_a * col_b,
                col_b < vec3<f32>(0.5),
            );
        }
        case 7u {  // Soft Light  (Pegtop formula)
            blended = (1.0 - 2.0 * col_b) * col_a * col_a + 2.0 * col_b * col_a;
        }
        case 8u {  // Hard Mix  (1 where A+B ≥ 1, else 0)
            blended = step(vec3<f32>(1.0), col_a + col_b);
        }
        case 9u {  // Screen  (inverse Multiply; brightens without hard clipping)
            blended = 1.0 - (1.0 - col_a) * (1.0 - col_b);
        }
        case 10u {  // Overlay  (Multiply if A<0.5, Screen if A≥0.5)
            blended = select(
                1.0 - 2.0 * (1.0 - col_a) * (1.0 - col_b),
                2.0 * col_a * col_b,
                col_a < vec3<f32>(0.5),
            );
        }
        case 11u {  // Color Dodge  (brightens A by B; B=1 → white)
            blended = select(
                min(vec3<f32>(1.0), col_a / max(1.0 - col_b, vec3<f32>(1e-6))),
                vec3<f32>(1.0),
                col_b >= vec3<f32>(1.0 - 1e-6),
            );
        }
        case 12u {  // Color Burn  (darkens A by B; B=0 → black)
            blended = select(
                1.0 - min(vec3<f32>(1.0), (1.0 - col_a) / max(col_b, vec3<f32>(1e-6))),
                vec3<f32>(0.0),
                col_b <= vec3<f32>(1e-6),
            );
        }
        case 13u {  // Difference
            blended = abs(col_a - col_b);
        }
        case 14u {  // Exclusion  (softer Difference)
            blended = col_a + col_b - 2.0 * col_a * col_b;
        }
        case 15u {  // Hue  (hue of B, saturation+luminosity of A)
            blended = blm_set_lum(blm_set_sat(col_b, blm_sat(col_a)), blm_lum(col_a));
        }
        case 16u {  // Saturation  (saturation of B, hue+luminosity of A)
            blended = blm_set_lum(blm_set_sat(col_a, blm_sat(col_b)), blm_lum(col_a));
        }
        case 17u {  // Color  (hue+saturation of B, luminosity of A)
            blended = blm_set_lum(col_b, blm_lum(col_a));
        }
        case 18u {  // Luminosity  (luminosity of B, hue+saturation of A)
            blended = blm_set_lum(col_a, blm_lum(col_b));
        }
        default {  // 0 = Add (and fallback for any unknown value)
            blended = clamp(col_a + col_b, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }

    // Fade by density_01: isolated sparse pixels approach background rather than ~30% brightness.
    let v = pow(clamp(blended * params.brightness, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / params.gamma));
    return vec4<f32>(v * grad_density_01, 1.0);
}
