use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::gradient::Gradient;
use crate::ui::{SolidShadingModel, UiState};
use gpu::{BlendMode, RenderMode};

/// Solid mode's material/lighting/reflection look -- the Solid-mode
/// equivalent of `gradient_a`/`gradient_b`/`blend_mode` for Light mode.
/// Deliberately excludes mesh density/geometry (point count, tube radius,
/// sides), which are shape fidelity settings, not coloring/rendering style.
#[derive(Clone, Serialize, Deserialize)]
pub struct SolidColorSettings {
    pub color:                [u8; 4],
    pub alpha:                f32,
    pub ambient:               f32,
    pub light_azimuth_deg:     f32,
    pub light_elevation_deg:   f32,
    pub specular_color:        [u8; 4],
    pub shininess:             f32,
    pub shading_model:         SolidShadingModel,
    pub metalness:             f32,
    pub anisotropy:            f32,
    pub toon_bands:            f32,
    pub toon_rim:              f32,
    pub sss_strength:          f32,
    pub sss_power:             f32,
    pub roughness:             f32,
    pub reflectivity:          f32,
    pub ior:                   f32,
    pub refraction:            f32,
    pub sky_top:               [u8; 4],
    pub sky_bottom:            [u8; 4],
}

impl Default for SolidColorSettings {
    fn default() -> Self {
        Self {
            color:              [220, 220, 225, 255],
            alpha:              1.0,
            ambient:            0.25,
            light_azimuth_deg:  35.0,
            light_elevation_deg: 55.0,
            specular_color:     [255, 255, 255, 255],
            shininess:          24.0,
            shading_model:      SolidShadingModel::BlinnPhong,
            metalness:          0.0,
            anisotropy:         0.0,
            toon_bands:         4.0,
            toon_rim:           0.3,
            sss_strength:       0.5,
            sss_power:          2.0,
            roughness:          0.3,
            reflectivity:       0.04,
            ior:                1.5,
            refraction:         0.0,
            sky_top:            [140, 180, 230, 255],
            sky_bottom:         [40, 40, 50, 255],
        }
    }
}

impl SolidColorSettings {
    pub fn capture(ui: &UiState) -> Self {
        let c = ui.solid_color;
        let spec = ui.solid_specular_color;
        let sky_top = ui.solid_sky_top;
        let sky_bottom = ui.solid_sky_bottom;
        Self {
            color:               [c.r(), c.g(), c.b(), c.a()],
            alpha:               ui.solid_alpha,
            ambient:             ui.solid_ambient,
            light_azimuth_deg:   ui.solid_light_azimuth_deg,
            light_elevation_deg: ui.solid_light_elevation_deg,
            specular_color:      [spec.r(), spec.g(), spec.b(), spec.a()],
            shininess:           ui.solid_shininess,
            shading_model:       ui.solid_shading_model,
            metalness:           ui.solid_metalness,
            anisotropy:          ui.solid_anisotropy,
            toon_bands:          ui.solid_toon_bands,
            toon_rim:            ui.solid_toon_rim,
            sss_strength:        ui.solid_sss_strength,
            sss_power:           ui.solid_sss_power,
            roughness:           ui.solid_roughness,
            reflectivity:        ui.solid_reflectivity,
            ior:                 ui.solid_ior,
            refraction:          ui.solid_refraction,
            sky_top:             [sky_top.r(), sky_top.g(), sky_top.b(), sky_top.a()],
            sky_bottom:          [sky_bottom.r(), sky_bottom.g(), sky_bottom.b(), sky_bottom.a()],
        }
    }

    pub fn apply(&self, ui: &mut UiState) {
        let [r, g, b, a] = self.color;
        ui.solid_color = egui::Color32::from_rgba_premultiplied(r, g, b, a);
        ui.solid_alpha = self.alpha;
        ui.solid_ambient = self.ambient;
        ui.solid_light_azimuth_deg = self.light_azimuth_deg;
        ui.solid_light_elevation_deg = self.light_elevation_deg;
        let [sr, sg, sb, sa] = self.specular_color;
        ui.solid_specular_color = egui::Color32::from_rgba_premultiplied(sr, sg, sb, sa);
        ui.solid_shininess = self.shininess;
        ui.solid_shading_model = self.shading_model;
        ui.solid_metalness = self.metalness;
        ui.solid_anisotropy = self.anisotropy;
        ui.solid_toon_bands = self.toon_bands;
        ui.solid_toon_rim = self.toon_rim;
        ui.solid_sss_strength = self.sss_strength;
        ui.solid_sss_power = self.sss_power;
        ui.solid_roughness = self.roughness;
        ui.solid_reflectivity = self.reflectivity;
        ui.solid_ior = self.ior;
        ui.solid_refraction = self.refraction;
        let [tr, tg, tb, ta] = self.sky_top;
        ui.solid_sky_top = egui::Color32::from_rgba_premultiplied(tr, tg, tb, ta);
        let [br, bg, bb, ba] = self.sky_bottom;
        ui.solid_sky_bottom = egui::Color32::from_rgba_premultiplied(br, bg, bb, ba);
    }
}

fn default_render_mode_for_legacy_color_sets() -> RenderMode {
    // Pre-Solid-mode saved color sets only ever applied to Light (gradients +
    // blend mode are Light-specific), so that's the non-disruptive default
    // when deserializing a color set saved before this field existed.
    RenderMode::Light
}

/// A named, reusable coloring scheme: everything that controls how the
/// attractor is colored and rendered, independent of its shape
/// (camera/attractor params are deliberately excluded). Includes which
/// render mode it was captured in -- applying a color set switches to that
/// mode, since e.g. a Solid-flavored material/lighting setup has no meaning
/// under Light's gradient/blend-mode coloring and vice versa.
#[derive(Clone, Serialize, Deserialize)]
pub struct ColorSet {
    pub name: String,
    #[serde(default = "default_render_mode_for_legacy_color_sets")]
    pub render_mode: RenderMode,
    pub color_space_srgb: bool,
    pub brightness: f32,
    pub gamma: f32,
    pub bg_color: [u8; 4],
    pub blend_mode: BlendMode,
    pub gradient_a: Gradient,
    pub gradient_b: Gradient,
    #[serde(default)]
    pub solid: SolidColorSettings,
}

impl ColorSet {
    pub fn capture(name: String, ui: &UiState) -> Self {
        let c = ui.bg_color;
        Self {
            name,
            render_mode: ui.render_mode,
            color_space_srgb: ui.color_space_srgb,
            brightness: ui.brightness,
            gamma: ui.gamma,
            bg_color: [c.r(), c.g(), c.b(), c.a()],
            blend_mode: ui.blend_mode,
            gradient_a: ui.gradient_a.clone(),
            gradient_b: ui.gradient_b.clone(),
            solid: SolidColorSettings::capture(ui),
        }
    }

    pub fn apply(&self, ui: &mut UiState) {
        ui.render_mode = self.render_mode;
        ui.dirty = true;
        ui.color_space_srgb = self.color_space_srgb;
        ui.brightness = self.brightness;
        ui.gamma = self.gamma;
        let [r, g, b, a] = self.bg_color;
        ui.bg_color = egui::Color32::from_rgba_premultiplied(r, g, b, a);
        ui.blend_mode = self.blend_mode;
        ui.gradient_a = self.gradient_a.clone();
        ui.gradient_b = self.gradient_b.clone();
        ui.gradient_a_dirty = true;
        ui.gradient_b_dirty = true;
        self.solid.apply(ui);
    }
}

/// Built-in color sets shipped with the app. Starter placeholder — add more
/// entries here once a few good-looking combinations are settled on.
pub fn built_in_color_sets() -> Vec<ColorSet> {
    vec![ColorSet {
        name: "Default".to_string(),
        render_mode: RenderMode::Light,
        color_space_srgb: true,
        brightness: 1.1,
        gamma: 0.95,
        bg_color: [0, 0, 0, 255],
        blend_mode: BlendMode::Add,
        gradient_a: Gradient::density_default(),
        gradient_b: Gradient::speed_default(),
        solid: SolidColorSettings::default(),
    }]
}

fn config_dir() -> PathBuf {
    if let Ok(appdata) = std::env::var("APPDATA") {
        return PathBuf::from(appdata).join("strange-attractor");
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg).join("strange-attractor");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".config").join("strange-attractor");
    }
    PathBuf::from(".")
}

fn color_sets_path() -> PathBuf {
    config_dir().join("color_sets.json")
}

/// Loads the user's custom color sets from the per-user config file. Returns
/// an empty list (rather than an error) if the file doesn't exist yet or
/// can't be parsed, since this runs unconditionally at startup.
pub fn load_custom_color_sets() -> Vec<ColorSet> {
    let path = color_sets_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

pub fn save_custom_color_sets(sets: &[ColorSet]) {
    let path = color_sets_path();
    if let Some(dir) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::error!("Failed to create color set config directory: {e}");
            return;
        }
    }
    match serde_json::to_string_pretty(sets) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                log::error!("Failed to save color sets: {e}");
            }
        }
        Err(e) => log::error!("Failed to serialize color sets: {e}"),
    }
}
