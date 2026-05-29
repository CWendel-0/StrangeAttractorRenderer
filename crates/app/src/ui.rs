use egui::Context;
use sim::attractor::ParamDesc;
use gpu::{BlendMode, RenderMode};

use crate::gradient::{Gradient, gradient_editor};

pub struct UiState {
    pub params:     Vec<f32>,
    pub brightness: f32,
    pub gamma:      f32,
    pub max_sigma:  f32,
    pub min_sigma:  f32,
    pub ss_scale:   u32,
    pub dirty:      bool,   // true when sim params changed this frame

    // ---- render mode ----
    pub render_mode: RenderMode,

    // ---- Light mode gradients + blend ----
    pub gradient_a:       Gradient,
    pub gradient_b:       Gradient,
    pub gradient_a_dirty: bool,
    pub gradient_b_dirty: bool,
    pub blend_mode:       BlendMode,

    // Selected stop index for each gradient editor (persisted across frames).
    selected_stop_a: Option<usize>,
    selected_stop_b: Option<usize>,
}

impl UiState {
    pub fn from_descriptors(descs: &[ParamDesc]) -> Self {
        Self {
            params:     descs.iter().map(|d| d.default).collect(),
            brightness: 1.0,
            gamma:      2.2,
            max_sigma:  1.5,
            min_sigma:  0.1,
            ss_scale:   2,
            dirty:      false,
            render_mode: RenderMode::Monochrome,
            gradient_a:       Gradient::density_default(),
            gradient_b:       Gradient::speed_default(),
            gradient_a_dirty: true,   // upload on first frame
            gradient_b_dirty: true,
            blend_mode:      BlendMode::Add,
            selected_stop_a: None,
            selected_stop_b: None,
        }
    }

    pub fn show(&mut self, ctx: &Context, descs: &[ParamDesc]) {
        self.dirty = false;

        egui::SidePanel::left("attractor_panel")
            .resizable(true)
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.heading("Lorenz");
                ui.separator();

                ui.label("Parameters");
                for (i, desc) in descs.iter().enumerate() {
                    let old = self.params[i];
                    ui.add(
                        egui::Slider::new(&mut self.params[i], desc.min..=desc.max)
                            .text(desc.name)
                            .clamping(egui::SliderClamping::Always),
                    );
                    if (self.params[i] - old).abs() > 1e-4 {
                        self.dirty = true;
                    }
                }

                ui.separator();
                ui.label("Rendering mode");
                ui.horizontal(|ui| {
                    let was = self.render_mode;
                    ui.selectable_value(&mut self.render_mode, RenderMode::Monochrome, "Monochrome");
                    ui.selectable_value(&mut self.render_mode, RenderMode::Light, "Light");
                    if self.render_mode != was {
                        // Switching mode clears the accumulation so the new colours show
                        // immediately instead of mixing with old mono data.
                        self.dirty = true;
                    }
                });

                ui.separator();
                ui.label("Display");
                ui.add(egui::Slider::new(&mut self.brightness, 0.1..=5.0).text("Brightness"));
                ui.add(egui::Slider::new(&mut self.gamma, 0.5..=4.0).text("Gamma"));

                ui.separator();
                ui.label("Anti-aliasing (supersampling)");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.ss_scale, 1u32, "1×");
                    ui.selectable_value(&mut self.ss_scale, 2u32, "2×");
                    ui.selectable_value(&mut self.ss_scale, 4u32, "4×");
                });

                ui.separator();
                ui.label("Blur (Density Estimation)");
                // Cap at 5.0: each DE kernel tap reads ss_scale² accum values, so
                // radius = ceil(sigma*3). At 4× SS with sigma=5 → 15 taps × 16 reads
                // per pixel; above that the DE passes become the frame-rate bottleneck.
                ui.add(
                    egui::Slider::new(&mut self.max_sigma, 0.1..=5.0)
                        .text("Max blur σ")
                        .clamping(egui::SliderClamping::Always),
                );
                ui.add(
                    egui::Slider::new(&mut self.min_sigma, 0.1..=3.0)
                        .text("Min blur σ")
                        .clamping(egui::SliderClamping::Always),
                );
                // Clamp after both sliders render so the user sees the constrained value.
                self.min_sigma = self.min_sigma.min(self.max_sigma);

                // ---- Light mode gradient editors ----
                if self.render_mode == RenderMode::Light {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Blend mode");
                        egui::ComboBox::from_id_salt("blend_mode")
                            .selected_text(self.blend_mode.label())
                            .show_ui(ui, |ui| {
                                for &mode in BlendMode::ALL {
                                    ui.selectable_value(&mut self.blend_mode, mode, mode.label());
                                }
                            });
                    });

                    ui.separator();
                    ui.label("Gradient A  (density)");
                    ui.add_space(2.0);
                    if gradient_editor(
                        ui,
                        &mut self.gradient_a,
                        &mut self.selected_stop_a,
                        egui::Id::new("grad_a"),
                    ) {
                        self.gradient_a_dirty = true;
                    }

                    ui.add_space(6.0);
                    ui.label("Gradient B  (speed)");
                    ui.add_space(2.0);
                    if gradient_editor(
                        ui,
                        &mut self.gradient_b,
                        &mut self.selected_stop_b,
                        egui::Id::new("grad_b"),
                    ) {
                        self.gradient_b_dirty = true;
                    }
                }
            });
    }
}
