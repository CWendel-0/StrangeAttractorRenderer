use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::gradient::Gradient;
use crate::ui::UiState;
use gpu::BlendMode;

/// A named, reusable coloring scheme: everything that controls how the
/// attractor is colored, independent of its shape (camera/attractor params
/// are deliberately excluded).
#[derive(Clone, Serialize, Deserialize)]
pub struct ColorSet {
    pub name: String,
    pub color_space_srgb: bool,
    pub brightness: f32,
    pub gamma: f32,
    pub bg_color: [u8; 4],
    pub blend_mode: BlendMode,
    pub gradient_a: Gradient,
    pub gradient_b: Gradient,
}

impl ColorSet {
    pub fn capture(name: String, ui: &UiState) -> Self {
        let c = ui.bg_color;
        Self {
            name,
            color_space_srgb: ui.color_space_srgb,
            brightness: ui.brightness,
            gamma: ui.gamma,
            bg_color: [c.r(), c.g(), c.b(), c.a()],
            blend_mode: ui.blend_mode,
            gradient_a: ui.gradient_a.clone(),
            gradient_b: ui.gradient_b.clone(),
        }
    }

    pub fn apply(&self, ui: &mut UiState) {
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
    }
}

/// Built-in color sets shipped with the app. Starter placeholder — add more
/// entries here once a few good-looking combinations are settled on.
pub fn built_in_color_sets() -> Vec<ColorSet> {
    vec![ColorSet {
        name: "Default".to_string(),
        color_space_srgb: true,
        brightness: 1.1,
        gamma: 0.95,
        bg_color: [0, 0, 0, 255],
        blend_mode: BlendMode::Add,
        gradient_a: Gradient::density_default(),
        gradient_b: Gradient::speed_default(),
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
