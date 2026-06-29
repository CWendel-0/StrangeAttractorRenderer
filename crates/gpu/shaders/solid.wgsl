// Solid render mode: rasterizes a lit tube mesh with a single shadow-mapped
// directional light. BRDF math lives in shading_common.wgsl (concatenated
// ahead of this file at shader-module creation time in solid.rs) as a pure
// function with no params/binding dependencies, also used by Points mode's
// composite pass -- but the params struct, bindings, and shadow mechanism
// here are Solid-specific (a real shadow-map depth texture), since Points
// mode's "shadow map" is a manual storage-buffer comparison instead.
// Alpha blending is always enabled in the main pipeline -- at alpha=1 this
// is indistinguishable from opaque rendering, so one pipeline covers both
// cases. The model transform is identity (mesh is already in world space),
// so no normal matrix is needed. Transparency is not depth-sorted, so
// overlapping tube strands may show order-dependent blending artifacts --
// an accepted v1 simplification.

struct SolidParams {
    view_proj:          mat4x4<f32>,
    light_view_proj:    mat4x4<f32>, // orthographic, framing the scene from the light's side
    camera_pos:         vec4<f32>,   // xyz = world-space eye position
    light_dir_ambient:  vec4<f32>,   // xyz = unit vector toward the light, w = ambient term
    base_color_alpha:   vec4<f32>,   // xyz = material color, w = opacity
    specular_shininess: vec4<f32>,   // xyz = specular/rim color, w = max shininess exponent (at roughness=0) -- Blinn-Phong/Oren-Nayar/Toon
    material_extra:     vec4<f32>,   // x = roughness [0,1], y = metalness [0,1] (Cook-Torrance/Anisotropic GGX), z = shading model id, w = anisotropy [-1,1] (Anisotropic GGX)
    reflect_refract:    vec4<f32>,   // x = reflectivity / dielectric F0, y = IOR, z = refraction strength, w unused
    sky_top:            vec4<f32>,   // xyz = sky color looking "up", w unused
    sky_bottom:         vec4<f32>,   // xyz = sky color looking "down", w unused
    model_params:       vec4<f32>,   // meaning depends on shading model -- see shading_common.wgsl
}

@group(0) @binding(0) var<uniform> params: SolidParams;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

const SHADOW_BIAS: f32 = 0.0015;

fn shadow_factor(world_pos: vec3<f32>) -> f32 {
    let light_clip = params.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;
    // NDC xy in [-1,1] -> texture uv in [0,1], flipping y (texture v grows
    // downward, NDC y grows upward).
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return 1.0; // outside the light's frustum (shouldn't happen, frustum is fit to the scene) -- assume lit
    }
    let depth = ndc.z - SHADOW_BIAS;

    // 3x3 PCF: average several comparison samples for a softer edge instead
    // of a single hard-edged tap.
    var sum = 0.0;
    let texel = 1.0 / 2048.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
        }
    }
    return sum / 9.0;
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) tangent:  vec3<f32>, // along the tube's length -- only used by Anisotropic GGX
}

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       world_pos:    vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       world_tangent: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = params.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_normal = in.normal;
    out.world_tangent = in.tangent;
    return out;
}

// Depth-only pass from the light's point of view -- no fragment stage.
@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
    return params.light_view_proj * vec4<f32>(in.position, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(params.light_dir_ambient.xyz);
    let v = normalize(params.camera_pos.xyz - in.world_pos);
    let h = normalize(l + v);
    let shadow = shadow_factor(in.world_pos);

    return shade(
        n, v, l, h, in.world_tangent,
        shadow,
        params.base_color_alpha.xyz, params.base_color_alpha.w,
        params.light_dir_ambient.w,
        params.material_extra.x, params.material_extra.y, params.material_extra.w, params.material_extra.z,
        params.specular_shininess.xyz, params.specular_shininess.w,
        params.reflect_refract.x, params.reflect_refract.y, params.reflect_refract.z,
        params.sky_top.xyz, params.sky_bottom.xyz,
        params.model_params,
    );
}
