// Points mode shading pass: runs once per *supersample* texel (ss_width x
// ss_height resolution, the same grid the sim shaders splat into), turning
// the bilaterally filled camera-space depth/hit buffers into a shaded color
// at that resolution. This is the key fix for jaggedness at self-overlap
// boundaries (where one strand of the attractor passes in front of another):
// each supersample texel reconstructs and shades its *own* nearest surface
// independently, so there's never a single winner-take-all depth pick
// blending two different surfaces' geometry together. points_composite.wgsl
// then downsamples this already-correctly-shaded buffer to canvas
// resolution by blending colors (not geometry), exactly like a classic
// supersampled antialiasing pass.
//
// Reconstructs a world-space position from the stored depth via the inverse
// view-projection matrix, and a normal from the screen-space derivative of
// that reconstructed position (`dpdx`/`dpdy` -- genuinely approximate, not a
// real geometric normal, same "fake 3D" premise as before). Shadows come
// from the light-space depth buffer (`points_light_depth`, written directly
// by the sim shaders) via a manual 3x3 PCF-style comparison.
//
// Lighting reuses the same `shade()` function as Solid mode
// (shading_common.wgsl, concatenated ahead of this file in points.rs).

struct PointsParams {
    view_proj:          mat4x4<f32>,
    inverse_view_proj:  mat4x4<f32>,
    light_view_proj:    mat4x4<f32>, // orthographic, framing the scene from the light's side
    camera_pos:         vec4<f32>,   // xyz = world-space eye position
    light_dir_ambient:  vec4<f32>,   // xyz = unit vector toward the light, w = ambient term
    base_color_alpha:   vec4<f32>,   // xyz = material color, w = opacity
    specular_shininess: vec4<f32>,   // xyz = specular/rim color, w = max shininess exponent (at roughness=0)
    material_extra:     vec4<f32>,   // x = roughness [0,1], y = metalness [0,1], z = shading model id, w = anisotropy [-1,1]
    reflect_refract:    vec4<f32>,   // x = reflectivity / dielectric F0, y = IOR, z = refraction strength, w unused
    sky_top:            vec4<f32>,   // xyz = sky color looking "up", w unused
    sky_bottom:         vec4<f32>,   // xyz = sky color looking "down", w unused
    model_params:       vec4<f32>,   // meaning depends on shading model -- see shading_common.wgsl
    // x = canvas width, y = canvas height, z = ss_scale, w = ss_width
    canvas_a:           vec4<u32>,
    // x = ss_height, y = light_buf_size, zw unused
    canvas_b:           vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: PointsParams;
@group(0) @binding(1) var<storage, read> points_depth_filled: array<u32>;
@group(0) @binding(2) var<storage, read> points_hit_filled: array<u32>;
@group(0) @binding(3) var<storage, read> points_light_depth: array<u32>;

const DEPTH_MAX: f32 = 4294967295.0;
const SHADOW_BIAS: f32 = 0.0015;

fn decode_depth(enc: u32) -> f32 {
    return 1.0 - f32(enc) / DEPTH_MAX;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

fn points_shadow_factor(world_pos: vec3<f32>) -> f32 {
    let light_buf_size = params.canvas_b.y;
    let light_clip = params.light_view_proj * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 {
        return 1.0;
    }
    let light_ndc = light_clip.xyz / light_clip.w;
    if light_ndc.x < -1.0 || light_ndc.x > 1.0 || light_ndc.y < -1.0 || light_ndc.y > 1.0 {
        return 1.0;
    }
    let lxi = i32((light_ndc.x * 0.5 + 0.5) * f32(light_buf_size));
    let lyi = i32((1.0 - (light_ndc.y * 0.5 + 0.5)) * f32(light_buf_size));

    var lit_sum = 0.0;
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sx = lxi + dx;
            let sy = lyi + dy;
            if sx < 0 || sy < 0 || u32(sx) >= light_buf_size || u32(sy) >= light_buf_size {
                continue;
            }
            let sidx = u32(sy) * light_buf_size + u32(sx);
            count += 1.0;
            let stored_enc = points_light_depth[sidx];
            if stored_enc == 0u {
                lit_sum += 1.0; // no recorded occluder at this texel
                continue;
            }
            let stored_z = decode_depth(stored_enc);
            if light_ndc.z <= stored_z + SHADOW_BIAS {
                lit_sum += 1.0; // this point is at/in front of the nearest recorded surface
            }
        }
    }
    if count <= 0.0 {
        return 1.0;
    }
    return lit_sum / count;
}

struct FsOut {
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> FsOut {
    let ss_width  = params.canvas_a.w;
    let ss_height = params.canvas_b.x;

    let sx = u32(frag_pos.x);
    let sy = u32(frag_pos.y);
    let idx = sy * ss_width + sx;

    let hit = points_hit_filled[idx];
    var out: FsOut;
    if hit == 0u {
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }

    let ndc_z = decode_depth(points_depth_filled[idx]);
    let ndc_x = (f32(sx) + 0.5) / f32(ss_width)  * 2.0 - 1.0;
    let ndc_y = 1.0 - (f32(sy) + 0.5) / f32(ss_height) * 2.0;

    let world_pos4 = params.inverse_view_proj * vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    let world_pos = world_pos4.xyz / world_pos4.w;

    // Fake normal: the screen-space derivative of the reconstructed
    // position, not a true geometric normal -- this is the whole "fake 3D"
    // premise of this render mode. Degenerates near silhouette/overlap edges
    // where a neighboring fragment in the same 2x2 quad has no data, or
    // belongs to a different surface; accepted v1 limitation.
    let normal_raw = cross(dpdx(world_pos), dpdy(world_pos));
    let normal_len = length(normal_raw);
    let n = select(vec3<f32>(0.0, 0.0, 1.0), normal_raw / max(normal_len, 1e-8), normal_len > 1e-8);

    let v = normalize(params.camera_pos.xyz - world_pos);
    let l = normalize(params.light_dir_ambient.xyz);
    let h = normalize(l + v);
    // Anisotropic GGX has no natural "grain" direction for a screen-space
    // fake surface; world-up gives a uniform (if arbitrary) highlight
    // orientation rather than a broken one -- same approach Solid mode's
    // billboard predecessor used.
    let tangent = vec3<f32>(0.0, 1.0, 0.0);

    let shadow = points_shadow_factor(world_pos);

    let shaded = shade(
        n, v, l, h, tangent,
        shadow,
        params.base_color_alpha.xyz, params.base_color_alpha.w,
        params.light_dir_ambient.w,
        params.material_extra.x, params.material_extra.y, params.material_extra.w, params.material_extra.z,
        params.specular_shininess.xyz, params.specular_shininess.w,
        params.reflect_refract.x, params.reflect_refract.y, params.reflect_refract.z,
        params.sky_top.xyz, params.sky_bottom.xyz,
        params.model_params,
    );
    out.color = vec4<f32>(shaded.rgb, shaded.a);
    return out;
}
