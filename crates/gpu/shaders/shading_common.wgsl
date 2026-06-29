// Shared lighting code for Solid mode's tube-mesh renderer (solid.wgsl) and
// Points mode's screen-space "fake 3D" composite pass (points_composite.wgsl).
// Deliberately self-contained (no struct/binding declarations, no global
// `params` access) since the two callers have genuinely different params
// structs and shadow mechanisms (a real depth texture for Solid, a manual
// storage-buffer comparison for Points) -- each caller computes its own
// shadow value and extracts its own material fields, then calls `shade()`
// with everything passed explicitly.

fn sky_color(dir: vec3<f32>, sky_top: vec3<f32>, sky_bottom: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(sky_bottom, sky_top, t);
}

const PI: f32 = 3.14159265359;

// Trowbridge-Reitz / GGX normal distribution (isotropic).
fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d  = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}

// Anisotropic GGX normal distribution: alpha_t/alpha_b stretch the highlight
// along the tangent/bitangent axes instead of being radially symmetric.
fn distribution_ggx_aniso(n_dot_h: f32, t_dot_h: f32, b_dot_h: f32, alpha_t: f32, alpha_b: f32) -> f32 {
    let term = (t_dot_h * t_dot_h) / (alpha_t * alpha_t) + (b_dot_h * b_dot_h) / (alpha_b * alpha_b) + n_dot_h * n_dot_h;
    return 1.0 / max(PI * alpha_t * alpha_b * term * term, 1e-6);
}

// Smith joint masking-shadowing term (Schlick-GGX approximation of each factor).
fn geometry_smith_ggx(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let k    = alpha * alpha * 0.5;
    let g_v  = n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6);
    let g_l  = n_dot_l / max(n_dot_l * (1.0 - k) + k, 1e-6);
    return g_v * g_l;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Oren-Nayar rough-diffuse term (qualitative form, Fresnel term omitted):
// flat Lambertian diffuse looks like smooth plastic; this accounts for
// microfacet self-shadowing/masking and reads more like chalk or unglazed
// ceramic as `sigma` (surface roughness, in radians) increases.
fn oren_nayar(n_dot_l: f32, n_dot_v: f32, l: vec3<f32>, v: vec3<f32>, n: vec3<f32>, sigma: f32) -> f32 {
    let sigma2 = sigma * sigma;
    let a = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
    let b = 0.45 * sigma2 / (sigma2 + 0.09);

    let theta_i = acos(clamp(n_dot_l, -1.0, 1.0));
    let theta_r = acos(clamp(n_dot_v, -1.0, 1.0));
    let alpha = max(theta_i, theta_r);
    let beta  = min(min(theta_i, theta_r), 1.55); // clamp away from PI/2 -- tan(beta) below blows up at grazing

    let l_perp = l - n * n_dot_l;
    let v_perp = v - n * n_dot_v;
    let l_proj = l_perp / max(length(l_perp), 1e-4);
    let v_proj = v_perp / max(length(v_perp), 1e-4);
    let cos_phi = max(dot(l_proj, v_proj), 0.0);

    return n_dot_l * (a + b * cos_phi * sin(alpha) * tan(beta));
}

// Computes the final lit+fresnel-blended color for a fragment. Fully
// explicit-argument (no global `params` access) so callers with different
// params structs and different shadow mechanisms can both use it:
// - n/v/l/h: normal/view/light/half vectors.
// - tangent: used only by Anisotropic GGX -- callers with no natural tangent
//   direction (e.g. Points mode's screen-space "fake" surfaces) can pass an
//   arbitrary fixed direction such as world-up; the highlight will look
//   uniformly "grained," not broken.
// - shadow: caller-computed shadow factor in [0,1] (1 = fully lit).
// - base/alpha: material color and opacity.
// - ambient: ambient term.
// - roughness/metalness/anisotropy/model: see each shading model's branch below.
// - specular_color/shininess: Blinn-Phong/Oren-Nayar/Toon only.
// - f0/ior/refraction_strength: reflection/refraction.
// - sky_top/sky_bottom: procedural sky for the fake reflection/refraction.
// - model_params: meaning depends on shading model (Toon: bands/rim strength;
//   Subsurface: translucency/glow tightness; unused otherwise).
fn shade(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, h: vec3<f32>, tangent: vec3<f32>,
    shadow: f32,
    base: vec3<f32>, alpha: f32,
    ambient: f32,
    roughness: f32, metalness: f32, anisotropy: f32, model: f32,
    specular_color: vec3<f32>, shininess: f32,
    f0: f32, ior_in: f32, refraction_strength_in: f32,
    sky_top: vec3<f32>, sky_bottom: vec3<f32>,
    model_params: vec4<f32>,
) -> vec4<f32> {
    let ior = max(ior_in, 1.001); // >1: entering a denser medium, eta=1/ior<1, so refract() never hits total internal reflection
    let sky_reflect = sky_color(reflect(-v, n), sky_top, sky_bottom);

    var lit: vec3<f32>;
    var fresnel: vec3<f32>;
    var refraction_strength: f32;

    if model < 0.5 {
        // Blinn-Phong: the original stylized model. Rougher surfaces scatter
        // light into a wider, dimmer-looking highlight -- this is what
        // carries a "frosted glass" look; ground glass isn't bumpy at a
        // visible scale, it's optically rough at a microscopic one, which
        // renderers model as exactly this, a broadened specular response,
        // not geometric detail. Floored at 3.0 rather than letting it reach
        // 1.0: shininess=1 reads as flat diffuse with no sheen left at all,
        // not "frosted."
        let diffuse = max(dot(n, l), 0.0) * shadow;
        let eff_shininess = mix(max(shininess, 1.0), 3.0, roughness);
        let spec = pow(max(dot(n, h), 0.0), eff_shininess) * shadow;

        lit = base * (ambient + (1.0 - ambient) * diffuse) + specular_color * spec;

        // Rougher surfaces reflect less like a mirror. Scaled by f0 itself
        // (not offset by a fixed Schlick base) so Reflectivity=0 means truly
        // zero reflection at every angle, including grazing -- otherwise
        // grazing angles always ramp toward full reflection regardless of
        // f0, amplifying the low-poly tube's per-facet normal-interpolation
        // error (sharpened by the 5th power) into visible dark bands toward
        // sky_bottom, one per facet.
        fresnel = vec3<f32>(clamp(f0 * (1.0 + 24.0 * pow(1.0 - max(dot(n, v), 0.0), 5.0)), 0.0, 1.0) * (1.0 - roughness));
        refraction_strength = refraction_strength_in;

    } else if model < 1.5 {
        // Cook-Torrance / GGX, full metal/dielectric workflow. `metalness`
        // blends between a dielectric (small, achromatic F0, diffuse-lit
        // albedo) and a metal (albedo itself is the F0 tint, no diffuse term).
        let f0_tinted = mix(vec3<f32>(f0), base, metalness);

        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 1e-4);
        let n_dot_h = max(dot(n, h), 0.0);
        let v_dot_h = max(dot(v, h), 0.0);
        let alpha   = max(roughness * roughness, 0.0009);

        let d_term = distribution_ggx(n_dot_h, alpha);
        let g_term = geometry_smith_ggx(n_dot_v, n_dot_l, alpha);
        let f_term = fresnel_schlick(v_dot_h, f0_tinted);

        let spec_color = (d_term * g_term * f_term) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
        let kd = base * (1.0 - metalness) * (vec3<f32>(1.0) - f_term);

        let ambient_color = base * (1.0 - metalness) * ambient;
        let direct_color  = (kd / PI + spec_color) * n_dot_l * shadow * (1.0 - ambient);
        lit = ambient_color + direct_color;

        // View-angle Fresnel for the fake sky reflection, tinted by F0 so
        // metals reflect in their own color rather than the sky's.
        fresnel = fresnel_schlick(n_dot_v, f0_tinted) * (1.0 - roughness);
        refraction_strength = refraction_strength_in * (1.0 - metalness); // metals don't transmit

    } else if model < 2.5 {
        // Oren-Nayar rough diffuse + a small Blinn-Phong specular kick on
        // top (pure Oren-Nayar with zero specular reads as completely flat).
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 1e-4);
        let sigma = roughness * (PI * 0.5); // roughness [0,1] -> sigma [0, PI/2] radians
        let diffuse = oren_nayar(n_dot_l, n_dot_v, l, v, n, sigma) * shadow;
        let eff_shininess = mix(max(shininess, 1.0), 3.0, roughness);
        let spec = pow(max(dot(n, h), 0.0), eff_shininess) * shadow;

        lit = base * (ambient + (1.0 - ambient) * diffuse) + specular_color * spec * 0.3;

        fresnel = vec3<f32>(clamp(f0 * (1.0 + 24.0 * pow(1.0 - n_dot_v, 5.0)), 0.0, 1.0) * (1.0 - roughness));
        refraction_strength = refraction_strength_in;

    } else if model < 3.5 {
        // Anisotropic GGX: stretches the specular highlight along the given
        // tangent direction instead of being radially symmetric -- looks
        // like brushed metal or satin running along that grain.
        let f0_tinted = mix(vec3<f32>(f0), base, metalness);

        let t_raw = tangent - n * dot(tangent, n); // re-orthogonalize against the normal
        let t = t_raw / max(length(t_raw), 1e-4);
        let b = cross(n, t);

        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 1e-4);
        let n_dot_h = max(dot(n, h), 0.0);
        let v_dot_h = max(dot(v, h), 0.0);
        let t_dot_h = dot(t, h);
        let b_dot_h = dot(b, h);

        let alpha   = max(roughness * roughness, 0.0009);
        let alpha_t = clamp(alpha * (1.0 + anisotropy), 0.0009, 1.0);
        let alpha_b = clamp(alpha * (1.0 - anisotropy), 0.0009, 1.0);

        let d_term = distribution_ggx_aniso(n_dot_h, t_dot_h, b_dot_h, alpha_t, alpha_b);
        let g_term = geometry_smith_ggx(n_dot_v, n_dot_l, sqrt(alpha_t * alpha_b));
        let f_term = fresnel_schlick(v_dot_h, f0_tinted);

        let spec_color = (d_term * g_term * f_term) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
        let kd = base * (1.0 - metalness) * (vec3<f32>(1.0) - f_term);

        let ambient_color = base * (1.0 - metalness) * ambient;
        let direct_color  = (kd / PI + spec_color) * n_dot_l * shadow * (1.0 - ambient);
        lit = ambient_color + direct_color;

        fresnel = fresnel_schlick(n_dot_v, f0_tinted) * (1.0 - roughness);
        refraction_strength = refraction_strength_in * (1.0 - metalness);

    } else if model < 4.5 {
        // Toon/cel shading: quantize the diffuse term into discrete bands
        // plus a rim light, instead of a continuous lighting response.
        let bands        = max(model_params.x, 1.0);
        let rim_strength = clamp(model_params.y, 0.0, 1.0);

        let n_dot_l = max(dot(n, l), 0.0) * shadow;
        let quant = floor(n_dot_l * bands) / bands;
        let toon_diffuse = mix(ambient, 1.0, quant);
        let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * rim_strength;

        lit = base * toon_diffuse + specular_color * rim;

        fresnel = vec3<f32>(clamp(f0 * (1.0 + 24.0 * pow(1.0 - max(dot(n, v), 0.0), 5.0)), 0.0, 1.0));
        refraction_strength = refraction_strength_in;

    } else {
        // Fake subsurface scattering: a "wrap" diffuse term that softens
        // the terminator (light falls off gradually past the horizon
        // instead of clipping at dot(n,l)=0) plus a view-aligned glow when
        // the light is roughly behind the surface from the camera's point
        // of view -- like light glowing through a thin waxy/frosted shell.
        let translucency = clamp(model_params.x, 0.0, 1.0);
        let sss_power    = max(model_params.y, 1.0);

        let wrap = 0.5;
        let wrapped_diffuse = max((dot(n, l) + wrap) / (1.0 + wrap), 0.0) * shadow;
        let transmission = pow(max(dot(v, -l), 0.0), sss_power) * translucency;

        lit = base * (ambient + (1.0 - ambient) * wrapped_diffuse) + base * transmission;

        fresnel = vec3<f32>(clamp(f0 * (1.0 + 24.0 * pow(1.0 - max(dot(n, v), 0.0), 5.0)), 0.0, 1.0) * (1.0 - roughness));
        refraction_strength = refraction_strength_in;
    }

    // Fake reflection/refraction: no real scene to sample, so both bounce
    // off a simple two-tone procedural sky instead of a real environment map.
    let sky_refract = sky_color(refract(-v, n, 1.0 / ior), sky_top, sky_bottom);
    let transmitted = mix(lit, sky_refract, refraction_strength);
    let final_rgb   = mix(transmitted, sky_reflect, clamp(fresnel, vec3<f32>(0.0), vec3<f32>(1.0)));

    return vec4<f32>(final_rgb, alpha);
}
