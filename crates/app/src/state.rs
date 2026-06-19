use serde::{Deserialize, Serialize};

use crate::camera::{ArcballCamera, CameraState};
use crate::gradient::Gradient;
use crate::ui::UiState;
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
}

const CURRENT_VERSION: u32 = 1;

impl SceneState {
    pub fn capture(ui: &UiState, camera: &ArcballCamera) -> Self {
        let c = ui.bg_color;
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
