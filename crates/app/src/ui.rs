use egui::Context;
use sim::{AttractorConfig, AttractorType, ParamKind};
use gpu::{BlendMode, RenderMode};

// C(O+3,3) = (O+1)(O+2)(O+3)/6 — monomials per equation for Sprott polynomial order O
fn sprott_num_terms(order: usize) -> usize {
    (order + 1) * (order + 2) * (order + 3) / 6
}

use crate::gradient::{Gradient, gradient_editor};
use crate::velocity_slider::velocity_slider;

pub struct UiState {
    pub attractor:  AttractorConfig,
    pub brightness: f32,
    pub gamma:      f32,
    pub max_sigma:  f32,
    pub min_sigma:  f32,
    pub ss_scale:   u32,
    pub dirty:          bool,
    pub type_changed:   bool,
    pub search_requested: bool,

    // ---- render mode ----
    pub render_mode: RenderMode,

    // ---- Light mode gradients + blend ----
    pub gradient_a:       Gradient,
    pub gradient_b:       Gradient,
    pub gradient_a_dirty: bool,
    pub gradient_b_dirty: bool,
    pub blend_mode:       BlendMode,

    selected_stop_a: Option<usize>,
    selected_stop_b: Option<usize>,

    attractor_open: bool,
    pub show_metrics: bool,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            attractor:  AttractorConfig::new(AttractorType::default()),
            brightness: 1.0,
            gamma:      2.2,
            max_sigma:  5.0,
            min_sigma:  0.3,
            ss_scale:   2,
            dirty:            false,
            type_changed:     false,
            search_requested: false,
            render_mode: RenderMode::Monochrome,
            gradient_a:       Gradient::density_default(),
            gradient_b:       Gradient::speed_default(),
            gradient_a_dirty: true,
            gradient_b_dirty: true,
            blend_mode:      BlendMode::Add,
            selected_stop_a: None,
            selected_stop_b: None,
            attractor_open: true,
            show_metrics: true,
        }
    }

    pub fn show(&mut self, ctx: &Context, searching: bool, metrics: Option<(f32, f32)>) {
        self.dirty = false;
        self.type_changed = false;
        self.search_requested = false;

        // ---- Floating attractor + parameters window ----
        egui::Window::new("Attractor")
            .open(&mut self.attractor_open)
            .resizable(true)
            .default_width(280.0)
            .default_height(500.0)
            .show(ctx, |ui| {
                ui.label("Attractor type");
                let prev_type = self.attractor.attractor_type;
                egui::ComboBox::from_id_salt("attractor_type")
                    .selected_text(self.attractor.attractor_type.label())
                    .show_ui(ui, |ui| {
                        for &t in AttractorType::ALL {
                            ui.selectable_value(&mut self.attractor.attractor_type, t, t.label());
                        }
                    });
                if self.attractor.attractor_type != prev_type {
                    self.attractor = AttractorConfig::new(self.attractor.attractor_type);
                    self.dirty = true;
                    self.type_changed = true;
                }

                ui.horizontal(|ui| {
                    let label = if searching { "Searching…" } else { "Randomize" };
                    if ui.add_enabled(!searching, egui::Button::new(label)).clicked() {
                        self.search_requested = true;
                    }
                });

                ui.separator();

                if self.attractor.attractor_type == AttractorType::PolySprott {
                    // Order spinner lives above the scroll so it stays visible while scrolling.
                    let mut order_val = self.attractor.params[0] as i32;
                    let order_changed = ui.horizontal(|ui| {
                        ui.checkbox(&mut self.attractor.frozen[0], "");
                        ui.label("Order");
                        ui.add(
                            egui::DragValue::new(&mut order_val)
                                .range(2..=5)
                                .speed(1.0),
                        ).changed()
                    }).inner;
                    if order_changed {
                        self.attractor.params[0] = order_val as f32;
                        self.dirty = true;
                    }

                    let n = sprott_num_terms(order_val as usize);
                    ui.label(format!("Parameters  ({} × 3)", n));

                    egui::ScrollArea::vertical()
                        .id_salt("params_scroll")
                        .show(ui, |ui| {
                            // X equation: params[1..=n]
                            for k in 0..n {
                                let label = format!("P{k}");
                                let idx   = 1 + k;
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut self.attractor.frozen[idx], "");
                                    if velocity_slider(
                                        ui,
                                        &mut self.attractor.params[idx],
                                        -1.5, 1.5,
                                        &label,
                                        egui::Id::new(("ps_x", k)),
                                    ) { self.dirty = true; }
                                });
                            }
                            // Y equation: params[57..=56+n]
                            for k in 0..n {
                                let label = format!("P{}", n + k);
                                let idx   = 57 + k;
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut self.attractor.frozen[idx], "");
                                    if velocity_slider(
                                        ui,
                                        &mut self.attractor.params[idx],
                                        -1.5, 1.5,
                                        &label,
                                        egui::Id::new(("ps_y", k)),
                                    ) { self.dirty = true; }
                                });
                            }
                            // Z equation: params[113..=112+n]
                            for k in 0..n {
                                let label = format!("P{}", 2 * n + k);
                                let idx   = 113 + k;
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut self.attractor.frozen[idx], "");
                                    if velocity_slider(
                                        ui,
                                        &mut self.attractor.params[idx],
                                        -1.5, 1.5,
                                        &label,
                                        egui::Id::new(("ps_z", k)),
                                    ) { self.dirty = true; }
                                });
                            }
                        });
                } else {
                    ui.label("Parameters");
                    egui::ScrollArea::vertical()
                        .id_salt("params_scroll")
                        .show(ui, |ui| {
                            let descs = self.attractor.descriptors();
                            for (i, desc) in descs.iter().enumerate() {
                                match &desc.kind {
                                    ParamKind::Continuous => {
                                        ui.horizontal(|ui| {
                                            ui.checkbox(&mut self.attractor.frozen[i], "");
                                            if velocity_slider(
                                                ui,
                                                &mut self.attractor.params[i],
                                                desc.min,
                                                desc.max,
                                                desc.name,
                                                egui::Id::new(("param_slider", i)),
                                            ) { self.dirty = true; }
                                        });
                                    }
                                    ParamKind::Integer => {
                                        let mut val = self.attractor.params[i] as i32;
                                        let changed = ui.horizontal(|ui| {
                                            ui.checkbox(&mut self.attractor.frozen[i], "");
                                            ui.label(desc.name);
                                            ui.add(
                                                egui::DragValue::new(&mut val)
                                                    .range(desc.min as i32..=desc.max as i32)
                                                    .speed(1.0),
                                            ).changed()
                                        }).inner;
                                        if changed {
                                            self.attractor.params[i] = val as f32;
                                            self.dirty = true;
                                        }
                                    }
                                    ParamKind::Enum(choices) => {
                                        let cur = self.attractor.params[i] as usize;
                                        let label = choices.get(cur).copied().unwrap_or("?");
                                        let mut idx = cur;
                                        ui.horizontal(|ui| {
                                            ui.checkbox(&mut self.attractor.frozen[i], "");
                                            egui::ComboBox::from_label(desc.name)
                                                .selected_text(label)
                                                .show_ui(ui, |ui| {
                                                    for (j, &ch) in choices.iter().enumerate() {
                                                        ui.selectable_value(&mut idx, j, ch);
                                                    }
                                                });
                                        });
                                        if idx != cur {
                                            self.attractor.params[i] = idx as f32;
                                            self.dirty = true;
                                        }
                                    }
                                }
                            }
                        });
                }
            });

        // ---- Sidebar: rendering / display controls only ----
        egui::SidePanel::left("render_panel")
            .resizable(true)
            .default_width(240.0)
            .show(ctx, |ui| {
                if !self.attractor_open {
                    if ui.button("Attractor…").clicked() {
                        self.attractor_open = true;
                    }
                    ui.separator();
                }

                ui.label("Rendering mode");
                ui.horizontal(|ui| {
                    let was = self.render_mode;
                    ui.selectable_value(&mut self.render_mode, RenderMode::Monochrome, "Monochrome");
                    ui.selectable_value(&mut self.render_mode, RenderMode::Light, "Colorful");
                    if self.render_mode != was {
                        self.dirty = true;
                    }
                });

                ui.separator();
                ui.label("Display");
                ui.add(egui::Slider::new(&mut self.brightness, 0.1..=5.0).text("Brightness"));
                ui.add(egui::Slider::new(&mut self.gamma, 0.5..=4.0).text("Gamma"));
                ui.checkbox(&mut self.show_metrics, "Show λ₁ / D_KY");

                ui.separator();
                ui.label("Anti-aliasing (supersampling)");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.ss_scale, 1u32, "1×");
                    ui.selectable_value(&mut self.ss_scale, 2u32, "2×");
                    ui.selectable_value(&mut self.ss_scale, 4u32, "4×");
                });

                ui.separator();
                ui.label("Blur (Density Estimation)");
                ui.add(
                    egui::Slider::new(&mut self.max_sigma, 0.5..=15.0)
                        .text("Max blur σ")
                        .clamping(egui::SliderClamping::Always),
                );
                ui.add(
                    egui::Slider::new(&mut self.min_sigma, 0.1..=5.0)
                        .text("Min blur σ")
                        .clamping(egui::SliderClamping::Always),
                );
                self.min_sigma = self.min_sigma.min(self.max_sigma);

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

        // ---- Bottom-left metrics overlay ----
        if self.show_metrics {
            egui::Area::new(egui::Id::new("metrics_overlay"))
                .anchor(egui::Align2::LEFT_BOTTOM, [12.0, -12.0])
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    egui::Frame::none()
                        .fill(egui::Color32::from_black_alpha(160))
                        .rounding(4.0)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            match metrics {
                                Some((ly1, dky)) if !ly1.is_nan() => {
                                    ui.label(
                                        egui::RichText::new(format!("λ₁  = {:.4}", ly1))
                                            .monospace()
                                            .color(egui::Color32::WHITE),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("D_KY = {:.4}", dky))
                                            .monospace()
                                            .color(egui::Color32::WHITE),
                                    );
                                }
                                Some(_) => {
                                    ui.label(
                                        egui::RichText::new("N/A")
                                            .monospace()
                                            .color(egui::Color32::from_gray(180)),
                                    );
                                }
                                None => {
                                    ui.label(
                                        egui::RichText::new("Computing…")
                                            .monospace()
                                            .color(egui::Color32::from_gray(180)),
                                    );
                                }
                            }
                        });
                });
        }
    }
}
