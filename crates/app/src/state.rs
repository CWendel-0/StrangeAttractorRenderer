use serde::{Deserialize, Serialize};

use crate::camera::{ArcballCamera, CameraState};
use crate::gradient::Gradient;
use crate::ui::{SolidShadingModel, UiState};
use gpu::{BlendMode, RenderMode};
use sim::AttractorConfig;

/// Everything needed to reproduce a render from scratch: attractor type and
/// parameters, all graphics/display settings, and camera position — but
/// deliberately not the accumulated histogram or rendered image, since the
/// point is to re-render the same scene, not restore a snapshot of pixels.
#[derive(Serialize, Deserialize)]
pub struct SceneState {
    version: u32,

    attractor: AttractorConfig,

    render_mode:       RenderMode,
    blend_mode:        BlendMode,
    color_space_srgb:  bool,

    brightness:      f32,
    gamma:           f32,
    max_sigma:       f32,
    min_sigma:       f32,
    alpha_power:     f32,
    noise_magnitude: f32,
    ss_scale:        u32,

    bg_color:   [u8; 4],
    gradient_a: Gradient,
    gradient_b: Gradient,

    camera: CameraState,

    // ---- Solid mode ----
    #[serde(default = "default_solid_point_count")]
    solid_point_count: u32,
    #[serde(default = "default_solid_tube_radius")]
    solid_tube_radius: f32,
    #[serde(default = "default_solid_sides")]
    solid_sides: u32,
    #[serde(default = "default_solid_color")]
    solid_color: [u8; 4],
    #[serde(default = "default_solid_ambient")]
    solid_ambient: f32,
    #[serde(default = "default_solid_alpha")]
    solid_alpha: f32,
    #[serde(default = "default_solid_light_azimuth_deg")]
    solid_light_azimuth_deg: f32,
    #[serde(default = "default_solid_light_elevation_deg")]
    solid_light_elevation_deg: f32,
    #[serde(default = "default_solid_specular_color")]
    solid_specular_color: [u8; 4],
    #[serde(default = "default_solid_shininess")]
    solid_shininess: f32,
    #[serde(default = "default_solid_shading_model")]
    solid_shading_model: SolidShadingModel,
    #[serde(default)]
    solid_metalness: f32,
    #[serde(default)]
    solid_anisotropy: f32,
    #[serde(default = "default_solid_toon_bands")]
    solid_toon_bands: f32,
    #[serde(default = "default_solid_toon_rim")]
    solid_toon_rim: f32,
    #[serde(default = "default_solid_sss_strength")]
    solid_sss_strength: f32,
    #[serde(default = "default_solid_sss_power")]
    solid_sss_power: f32,
    #[serde(default = "default_solid_roughness")]
    solid_roughness: f32,
    #[serde(default = "default_solid_reflectivity")]
    solid_reflectivity: f32,
    #[serde(default = "default_solid_ior")]
    solid_ior: f32,
    #[serde(default)]
    solid_refraction: f32,
    #[serde(default = "default_solid_sky_top")]
    solid_sky_top: [u8; 4],
    #[serde(default = "default_solid_sky_bottom")]
    solid_sky_bottom: [u8; 4],

    // ---- Points mode ----
    #[serde(default)]
    points_radius: u32,
}

fn default_solid_point_count() -> u32 { 100_000 }
fn default_solid_tube_radius() -> f32 { 0.05 }
fn default_solid_sides() -> u32 { 8 }
fn default_solid_color() -> [u8; 4] { [220, 220, 225, 255] }
fn default_solid_ambient() -> f32 { 0.25 }
fn default_solid_alpha() -> f32 { 1.0 }
fn default_solid_light_azimuth_deg() -> f32 { 35.0 }
fn default_solid_light_elevation_deg() -> f32 { 55.0 }
fn default_solid_specular_color() -> [u8; 4] { [255, 255, 255, 255] }
fn default_solid_shininess() -> f32 { 24.0 }
fn default_solid_shading_model() -> SolidShadingModel { SolidShadingModel::BlinnPhong }
fn default_solid_toon_bands() -> f32 { 4.0 }
fn default_solid_toon_rim() -> f32 { 0.3 }
fn default_solid_sss_strength() -> f32 { 0.5 }
fn default_solid_sss_power() -> f32 { 2.0 }
fn default_solid_roughness() -> f32 { 0.3 }
fn default_solid_reflectivity() -> f32 { 0.04 }
fn default_solid_ior() -> f32 { 1.5 }
fn default_solid_sky_top() -> [u8; 4] { [140, 180, 230, 255] }
fn default_solid_sky_bottom() -> [u8; 4] { [40, 40, 50, 255] }
const CURRENT_VERSION: u32 = 1;

impl SceneState {
    pub fn capture(ui: &UiState, camera: &ArcballCamera) -> Self {
        let c = ui.bg_color;
        let sc = ui.solid_color;
        let spec = ui.solid_specular_color;
        let sky_top = ui.solid_sky_top;
        let sky_bottom = ui.solid_sky_bottom;
        Self {
            version: CURRENT_VERSION,
            attractor: ui.attractor.clone(),
            render_mode: ui.render_mode,
            blend_mode: ui.blend_mode,
            color_space_srgb: ui.color_space_srgb,
            brightness: ui.brightness,
            gamma: ui.gamma,
            max_sigma: ui.max_sigma,
            min_sigma: ui.min_sigma,
            alpha_power: ui.alpha_power,
            noise_magnitude: ui.noise_magnitude,
            ss_scale: ui.ss_scale,
            bg_color: [c.r(), c.g(), c.b(), c.a()],
            gradient_a: ui.gradient_a.clone(),
            gradient_b: ui.gradient_b.clone(),
            camera: camera.to_state(),
            solid_point_count: ui.solid_point_count,
            solid_tube_radius: ui.solid_tube_radius,
            solid_sides: ui.solid_sides,
            solid_color: [sc.r(), sc.g(), sc.b(), sc.a()],
            solid_ambient: ui.solid_ambient,
            solid_alpha: ui.solid_alpha,
            solid_light_azimuth_deg: ui.solid_light_azimuth_deg,
            solid_light_elevation_deg: ui.solid_light_elevation_deg,
            solid_specular_color: [spec.r(), spec.g(), spec.b(), spec.a()],
            solid_shininess: ui.solid_shininess,
            solid_shading_model: ui.solid_shading_model,
            solid_metalness: ui.solid_metalness,
            solid_anisotropy: ui.solid_anisotropy,
            solid_toon_bands: ui.solid_toon_bands,
            solid_toon_rim: ui.solid_toon_rim,
            solid_sss_strength: ui.solid_sss_strength,
            solid_sss_power: ui.solid_sss_power,
            solid_roughness: ui.solid_roughness,
            solid_reflectivity: ui.solid_reflectivity,
            solid_ior: ui.solid_ior,
            solid_refraction: ui.solid_refraction,
            solid_sky_top: [sky_top.r(), sky_top.g(), sky_top.b(), sky_top.a()],
            solid_sky_bottom: [sky_bottom.r(), sky_bottom.g(), sky_bottom.b(), sky_bottom.a()],
            points_radius: ui.points_radius,
        }
    }

    /// Apply this state to the live UI and camera. Marks the UI dirty so the
    /// caller knows to reset the sim trajectories and clear the histogram —
    /// loading a state re-renders from scratch, it doesn't restore pixels.
    pub fn apply(&self, ui: &mut UiState, camera: &mut ArcballCamera) {
        ui.attractor = self.attractor.clone();
        ui.render_mode = self.render_mode;
        ui.blend_mode = self.blend_mode;
        ui.color_space_srgb = self.color_space_srgb;
        ui.brightness = self.brightness;
        ui.gamma = self.gamma;
        ui.max_sigma = self.max_sigma;
        ui.min_sigma = self.min_sigma;
        ui.alpha_power = self.alpha_power;
        ui.noise_magnitude = self.noise_magnitude;
        ui.ss_scale = self.ss_scale;
        let [r, g, b, a] = self.bg_color;
        ui.bg_color = egui::Color32::from_rgba_premultiplied(r, g, b, a);
        ui.gradient_a = self.gradient_a.clone();
        ui.gradient_b = self.gradient_b.clone();
        ui.gradient_a_dirty = true;
        ui.gradient_b_dirty = true;
        ui.solid_point_count = self.solid_point_count;
        ui.solid_tube_radius = self.solid_tube_radius;
        ui.solid_sides = self.solid_sides;
        let [sr, sg, sb, sa] = self.solid_color;
        ui.solid_color = egui::Color32::from_rgba_premultiplied(sr, sg, sb, sa);
        ui.solid_ambient = self.solid_ambient;
        ui.solid_alpha = self.solid_alpha;
        ui.solid_light_azimuth_deg = self.solid_light_azimuth_deg;
        ui.solid_light_elevation_deg = self.solid_light_elevation_deg;
        let [pr, pg, pb, pa] = self.solid_specular_color;
        ui.solid_specular_color = egui::Color32::from_rgba_premultiplied(pr, pg, pb, pa);
        ui.solid_shininess = self.solid_shininess;
        ui.solid_shading_model = self.solid_shading_model;
        ui.solid_metalness = self.solid_metalness;
        ui.solid_anisotropy = self.solid_anisotropy;
        ui.solid_toon_bands = self.solid_toon_bands;
        ui.solid_toon_rim = self.solid_toon_rim;
        ui.solid_sss_strength = self.solid_sss_strength;
        ui.solid_sss_power = self.solid_sss_power;
        ui.solid_roughness = self.solid_roughness;
        ui.solid_reflectivity = self.solid_reflectivity;
        ui.solid_ior = self.solid_ior;
        ui.solid_refraction = self.solid_refraction;
        let [str_, stg, stb, sta] = self.solid_sky_top;
        ui.solid_sky_top = egui::Color32::from_rgba_premultiplied(str_, stg, stb, sta);
        let [sbr, sbg, sbb, sba] = self.solid_sky_bottom;
        ui.solid_sky_bottom = egui::Color32::from_rgba_premultiplied(sbr, sbg, sbb, sba);
        ui.mesh_dirty_solid = true;
        ui.points_radius = self.points_radius;
        camera.apply_state(&self.camera);
    }

    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self).map_err(std::io::Error::from)
    }

    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_json::from_reader(file).map_err(std::io::Error::from)
    }
}

/// Checks that every keyframe shares the discrete settings that can't be
/// interpolated (attractor type, render/blend mode, color space, AA level).
/// Everything else (camera, gradients, attractor params, brightness, etc.)
/// is fair game for [`interpolate`].
pub(crate) fn validate_compatible(states: &[SceneState]) -> Result<(), String> {
    let first = &states[0];
    for (i, s) in states.iter().enumerate().skip(1) {
        if s.attractor.attractor_type != first.attractor.attractor_type {
            return Err(format!(
                "Keyframe {i} has attractor type {:?}, but keyframe 0 has {:?} — all keyframes must use the same attractor type.",
                s.attractor.attractor_type, first.attractor.attractor_type
            ));
        }
        if s.render_mode != first.render_mode {
            return Err(format!("Keyframe {i} has a different render mode than keyframe 0."));
        }
        if s.blend_mode != first.blend_mode {
            return Err(format!("Keyframe {i} has a different blend mode than keyframe 0."));
        }
        if s.color_space_srgb != first.color_space_srgb {
            return Err(format!("Keyframe {i} has a different color space than keyframe 0."));
        }
        if s.ss_scale != first.ss_scale {
            return Err(format!("Keyframe {i} has a different anti-aliasing (supersampling) setting than keyframe 0."));
        }
        if first.render_mode == RenderMode::Solid {
            // Mesh density and the BRDF itself are structural, not
            // continuously animatable -- a sudden jump partway through a
            // movie would look like popping, and there's no sensible
            // halfway point between e.g. Blinn-Phong and Toon shading.
            // Everything else (color, lighting, roughness, tube radius,
            // sky colors, ...) is fair game for `interpolate`.
            if s.solid_point_count != first.solid_point_count {
                return Err(format!("Keyframe {i} has a different Solid trajectory point count than keyframe 0."));
            }
            if s.solid_sides != first.solid_sides {
                return Err(format!("Keyframe {i} has a different Solid tube side count than keyframe 0."));
            }
            if s.solid_shading_model != first.solid_shading_model {
                return Err(format!("Keyframe {i} has a different Solid shading model than keyframe 0."));
            }
        }
        if first.render_mode == RenderMode::Points && s.solid_shading_model != first.solid_shading_model {
            // Shared with Solid mode for the same reason given above --
            // there's no sensible halfway point between two BRDFs.
            return Err(format!("Keyframe {i} has a different Points shading model than keyframe 0."));
        }
    }
    Ok(())
}

fn lerp_bg_color(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let mut out = [0u8; 4];
    for i in 0..4 {
        out[i] = (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Interpolate everything that's safe to interpolate between two compatible
/// keyframes (see [`validate_compatible`]); discrete fields are copied from
/// `a` since they're guaranteed equal across all keyframes.
pub(crate) fn interpolate(a: &SceneState, b: &SceneState, t: f32) -> SceneState {
    let t = t.clamp(0.0, 1.0);
    SceneState {
        version: CURRENT_VERSION,
        attractor: AttractorConfig {
            attractor_type: a.attractor.attractor_type,
            params: a.attractor.params.iter().zip(&b.attractor.params)
                .map(|(pa, pb)| pa + (pb - pa) * t)
                .collect(),
            frozen: a.attractor.frozen.clone(),
        },
        render_mode: a.render_mode,
        blend_mode: a.blend_mode,
        color_space_srgb: a.color_space_srgb,
        brightness: a.brightness + (b.brightness - a.brightness) * t,
        gamma: a.gamma + (b.gamma - a.gamma) * t,
        max_sigma: a.max_sigma + (b.max_sigma - a.max_sigma) * t,
        min_sigma: a.min_sigma + (b.min_sigma - a.min_sigma) * t,
        alpha_power: a.alpha_power + (b.alpha_power - a.alpha_power) * t,
        noise_magnitude: a.noise_magnitude + (b.noise_magnitude - a.noise_magnitude) * t,
        ss_scale: a.ss_scale,
        bg_color: lerp_bg_color(a.bg_color, b.bg_color, t),
        gradient_a: Gradient::lerp(&a.gradient_a, &b.gradient_a, t),
        gradient_b: Gradient::lerp(&a.gradient_b, &b.gradient_b, t),
        camera: CameraState::lerp(&a.camera, &b.camera, t),
        // Mesh density/BRDF are structural -- validate_compatible() requires
        // them equal across keyframes when render_mode is Solid, so copying
        // from `a` is exact, not an approximation. Everything else below is
        // a genuine continuous lerp.
        solid_point_count: a.solid_point_count,
        solid_tube_radius: a.solid_tube_radius + (b.solid_tube_radius - a.solid_tube_radius) * t,
        solid_sides: a.solid_sides,
        solid_color: lerp_bg_color(a.solid_color, b.solid_color, t),
        solid_ambient: a.solid_ambient + (b.solid_ambient - a.solid_ambient) * t,
        solid_alpha: a.solid_alpha + (b.solid_alpha - a.solid_alpha) * t,
        solid_light_azimuth_deg: a.solid_light_azimuth_deg + (b.solid_light_azimuth_deg - a.solid_light_azimuth_deg) * t,
        solid_light_elevation_deg: a.solid_light_elevation_deg + (b.solid_light_elevation_deg - a.solid_light_elevation_deg) * t,
        solid_specular_color: lerp_bg_color(a.solid_specular_color, b.solid_specular_color, t),
        solid_shininess: a.solid_shininess + (b.solid_shininess - a.solid_shininess) * t,
        solid_shading_model: a.solid_shading_model,
        solid_metalness: a.solid_metalness + (b.solid_metalness - a.solid_metalness) * t,
        solid_anisotropy: a.solid_anisotropy + (b.solid_anisotropy - a.solid_anisotropy) * t,
        solid_toon_bands: a.solid_toon_bands + (b.solid_toon_bands - a.solid_toon_bands) * t,
        solid_toon_rim: a.solid_toon_rim + (b.solid_toon_rim - a.solid_toon_rim) * t,
        solid_sss_strength: a.solid_sss_strength + (b.solid_sss_strength - a.solid_sss_strength) * t,
        solid_sss_power: a.solid_sss_power + (b.solid_sss_power - a.solid_sss_power) * t,
        solid_roughness: a.solid_roughness + (b.solid_roughness - a.solid_roughness) * t,
        solid_reflectivity: a.solid_reflectivity + (b.solid_reflectivity - a.solid_reflectivity) * t,
        solid_ior: a.solid_ior + (b.solid_ior - a.solid_ior) * t,
        solid_refraction: a.solid_refraction + (b.solid_refraction - a.solid_refraction) * t,
        solid_sky_top: lerp_bg_color(a.solid_sky_top, b.solid_sky_top, t),
        solid_sky_bottom: lerp_bg_color(a.solid_sky_bottom, b.solid_sky_bottom, t),
        // A small discrete footprint setting, not worth animating smoothly --
        // copied from `a`, same treatment as ss_scale above.
        points_radius: a.points_radius,
    }
}
