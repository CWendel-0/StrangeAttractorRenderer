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
    }
}
