// Points mode downsample pass: turns the per-supersample-texel shaded
// buffer (written by points_shade.wgsl, at ss_width x ss_height) into the
// final canvas-resolution image. Each supersample texel was already shaded
// against its own correct surface (no winner-take-all depth picking), so
// this pass only blends *colors* -- exactly the classic supersampled
// antialiasing downsample, and the reason self-overlap boundaries (where
// one strand passes in front of another) come out smooth instead of a hard
// seam: both sides of that boundary were shaded independently and correctly
// before this pass ever averages them together.
//
// Samples a Gaussian-weighted neighborhood that overlaps slightly into
// adjacent canvas pixels' supersample blocks (not just the exact
// non-overlapping ss_scale block) -- same reconstruction-filter rationale
// as de_h.wgsl's block_density -- so silhouette edges fade smoothly rather
// than flipping at a hard block boundary.
//
// Blends in premultiplied-alpha space: each shaded texel's color is already
// non-premultiplied opaque material color, with alpha 0 for "no data" and
// the material's own alpha otherwise, so premultiplying before the weighted
// sum and un-premultiplying after gives the correct color for partially
// covered output pixels (mixed empty/covered supersample texels) without
// empty texels' (irrelevant, possibly zero) color leaking in.

struct CompositeParams {
    // x = canvas width, y = canvas height, z = ss_scale, w = ss_width
    canvas_a: vec4<u32>,
    // x = ss_height, yzw unused
    canvas_b: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: CompositeParams;
@group(0) @binding(1) var points_shaded_tex: texture_2d<f32>;

const AA_SIGMA_SCALE: f32 = 0.5; // sigma, in supersample-texel units, relative to ss_scale

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ss_scale  = params.canvas_a.z;
    let ss_width  = params.canvas_a.w;
    let ss_height = params.canvas_b.x;

    let px = u32(frag_pos.x);
    let py = u32(frag_pos.y);

    let ss = f32(ss_scale);
    let cx = (f32(px) + 0.5) * ss;
    let cy = (f32(py) + 0.5) * ss;
    let sigma  = max(ss * AA_SIGMA_SCALE, 0.6);
    let inv2s2 = 0.5 / (sigma * sigma);
    let radius = i32(ceil(sigma * 1.5));
    let icx = i32(floor(cx));
    let icy = i32(floor(cy));

    var color_sum  = vec3<f32>(0.0);
    var alpha_sum  = 0.0; // premultiplied-alpha-weighted
    var weight_sum = 0.0;
    for (var dy = -radius; dy <= radius; dy++) {
        let sy = icy + dy;
        if sy < 0 || u32(sy) >= ss_height { continue; }
        let ddy = f32(sy) + 0.5 - cy;
        for (var dx = -radius; dx <= radius; dx++) {
            let sx = icx + dx;
            if sx < 0 || u32(sx) >= ss_width { continue; }
            let ddx = f32(sx) + 0.5 - cx;
            let w = exp(-(ddx * ddx + ddy * ddy) * inv2s2);
            let texel = textureLoad(points_shaded_tex, vec2<i32>(sx, sy), 0);
            color_sum  += w * texel.rgb * texel.a;
            alpha_sum  += w * texel.a;
            weight_sum += w;
        }
    }

    let out_alpha = clamp(alpha_sum / max(weight_sum, 1e-6), 0.0, 1.0);
    if out_alpha <= 1e-4 {
        discard;
    }
    let out_color = color_sum / max(alpha_sum, 1e-6);
    return vec4<f32>(out_color, out_alpha);
}
