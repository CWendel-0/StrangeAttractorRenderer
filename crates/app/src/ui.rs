use egui::Context;
use sim::{AttractorConfig, AttractorType, ParamKind};
use gpu::{BlendMode, RenderMode};

// C(O+3,3) = (O+1)(O+2)(O+3)/6 — monomials per equation for Sprott polynomial order O
fn sprott_num_terms(order: usize) -> usize {
    (order + 1) * (order + 2) * (order + 3) / 6
}

use crate::gradient::{Gradient, gradient_editor};
use crate::movie::{MovieStatus, OutputKind};
use crate::velocity_slider::velocity_slider;

pub struct UiState {
    pub attractor:  AttractorConfig,
    pub brightness: f32,
    pub gamma:      f32,
    pub max_sigma:   f32,
    pub min_sigma:   f32,
    pub alpha_power:      f32,
    pub noise_magnitude:  f32,
    pub ss_scale:   u32,
    pub dirty:          bool,
    pub type_changed:   bool,
    pub search_requested: bool,
    pub save_requested:   bool,
    pub save_state_requested: bool,
    pub load_state_requested: bool,
    pub color_space_srgb: bool,

    pub canvas_width:  u32,
    pub canvas_height: u32,
    pub canvas_dirty:  bool,
    pub expanded:      bool,

    // Camera input forwarded from the canvas widget each frame.
    pub viewport_drag_left:   egui::Vec2,
    pub viewport_drag_middle: egui::Vec2,
    pub viewport_scroll:      f32,

    pub bg_color: egui::Color32,

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
    pub iter_count:   u64,

    // ---- Movie render dialog ----
    pub movie_dialog_open:       bool,
    pub movie_keyframe_paths:    Vec<std::path::PathBuf>,
    pub movie_frames_per_step:   u32,
    pub movie_iters_per_frame:   u64,
    pub movie_loop_back:         bool,
    pub movie_output_kind:       OutputKind,
    pub movie_output_path:       Option<std::path::PathBuf>,
    pub movie_fps:                u32,
    pub movie_mp4_crf:            u8,
    pub movie_render_requested:  bool,
    pub movie_cancel_requested:  bool,
    pub movie_close_requested:   bool,
    pub movie_job_active:        bool,
    pub movie_status_for_ui:     Option<MovieStatus>,
}

impl UiState {
    pub fn new(canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            attractor:  AttractorConfig::new(AttractorType::default()),
            brightness: 1.1,
            gamma:      0.95,
            max_sigma:   1.0,
            min_sigma:   0.3,
            alpha_power:     2.0,
            noise_magnitude: 0.0,
            ss_scale:   2,
            dirty:            false,
            type_changed:     false,
            search_requested: false,
            save_requested:   false,
            save_state_requested: false,
            load_state_requested: false,
            color_space_srgb: true,
            canvas_width,
            canvas_height,
            canvas_dirty: false,
            expanded:     true,
            viewport_drag_left:   egui::Vec2::ZERO,
            viewport_drag_middle: egui::Vec2::ZERO,
            viewport_scroll:      0.0,
            bg_color: egui::Color32::BLACK,
            render_mode: RenderMode::Light,
            gradient_a:       Gradient::density_default(),
            gradient_b:       Gradient::speed_default(),
            gradient_a_dirty: true,
            gradient_b_dirty: true,
            blend_mode:      BlendMode::Add,
            selected_stop_a: None,
            selected_stop_b: None,
            attractor_open: true,
            show_metrics: true,
            iter_count:   0,

            movie_dialog_open:      false,
            movie_keyframe_paths:   Vec::new(),
            movie_frames_per_step:  30,
            movie_iters_per_frame:  50_000_000,
            movie_loop_back:        false,
            movie_output_kind:      OutputKind::PngSequence,
            movie_output_path:      None,
            movie_fps:              30,
            movie_mp4_crf:          18,
            movie_render_requested: false,
            movie_cancel_requested: false,
            movie_close_requested:  false,
            movie_job_active:       false,
            movie_status_for_ui:    None,
        }
    }

    pub fn show(&mut self, ctx: &Context, searching: bool, metrics: Option<(f32, f32)>, canvas_tex_id: Option<egui::TextureId>) {
        self.dirty = false;
        self.type_changed = false;
        self.search_requested = false;
        self.save_requested   = false;
        self.save_state_requested = false;
        self.load_state_requested = false;
        self.movie_render_requested = false;
        self.movie_cancel_requested = false;
        self.movie_close_requested  = false;
        self.viewport_drag_left   = egui::Vec2::ZERO;
        self.viewport_drag_middle = egui::Vec2::ZERO;
        self.viewport_scroll      = 0.0;

        // ---- Menu bar ----
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Save Image…").clicked() {
                        self.save_requested = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Save State…").clicked() {
                        self.save_state_requested = true;
                        ui.close_menu();
                    }
                    if ui.button("Load State…").clicked() {
                        self.load_state_requested = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Render Movie…").clicked() {
                        self.movie_dialog_open = true;
                        ui.close_menu();
                    }
                });
            });
        });

        self.show_movie_dialog(ctx);

        // ---- Floating attractor + parameters window ----
        if !self.movie_job_active {
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
        }

        // ---- Sidebar: rendering / display controls only ----
        if !self.movie_job_active {
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

                ui.label("Color space");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.color_space_srgb, true,  "sRGB");
                    ui.selectable_value(&mut self.color_space_srgb, false, "Linear");
                });

                ui.separator();
                ui.label("Canvas size");
                ui.horizontal(|ui| {
                    let mut cw = self.canvas_width as i32;
                    let mut ch = self.canvas_height as i32;
                    let wc = ui.add_enabled(!self.expanded, egui::DragValue::new(&mut cw).range(64..=4096).speed(1.0)).changed();
                    ui.label("×");
                    let hc = ui.add_enabled(!self.expanded, egui::DragValue::new(&mut ch).range(64..=4096).speed(1.0)).changed();
                    if wc || hc {
                        self.canvas_width  = (cw as u32).max(64);
                        self.canvas_height = (ch as u32).max(64);
                        self.canvas_dirty  = true;
                    }
                    let expand_label = if self.expanded { "Collapse" } else { "Expand" };
                    if ui.button(expand_label).clicked() {
                        self.expanded = !self.expanded;
                    }
                });

                ui.separator();
                ui.label("Display");
                ui.add(egui::Slider::new(&mut self.brightness, 0.1..=5.0).text("Brightness"));
                ui.add(egui::Slider::new(&mut self.gamma, 0.0..=4.0).text("Gamma"));
                ui.horizontal(|ui| {
                    ui.label("Background");
                    ui.color_edit_button_srgba(&mut self.bg_color);
                });
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
                    egui::Slider::new(&mut self.max_sigma, 0.0..=5.0)
                        .text("Max blur σ")
                        .clamping(egui::SliderClamping::Always),
                );
                ui.add(
                    egui::Slider::new(&mut self.min_sigma, 0.0..=2.0)
                        .text("Min blur σ")
                        .clamping(egui::SliderClamping::Always),
                );
                self.min_sigma = self.min_sigma.min(self.max_sigma);
                ui.add(
                    egui::Slider::new(&mut self.alpha_power, 1.0..=10.0)
                        .text("Alpha power")
                        .clamping(egui::SliderClamping::Always),
                );
                ui.add(
                    egui::Slider::new(&mut self.noise_magnitude, 0.0..=3.0)
                        .text("Noise (px)")
                        .clamping(egui::SliderClamping::Always),
                );

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

        // ---- Floating canvas window ----
        if let Some(tex_id) = canvas_tex_id {
            // When expanded, the canvas fills the area not claimed by panels.
            // ctx.available_rect() here is post-side-panel, so it's the central area.
            if self.expanded {
                let avail = ctx.available_rect();
                // Account for window title bar (~20px) and inner margin (~6px each side).
                let new_w = (avail.width()  - 12.0).max(64.0) as u32;
                let new_h = (avail.height() - 32.0).max(64.0) as u32;
                if new_w != self.canvas_width || new_h != self.canvas_height {
                    self.canvas_width  = new_w;
                    self.canvas_height = new_h;
                    self.canvas_dirty  = true;
                }
            }

            let base_window = egui::Window::new("Canvas")
                .default_pos(egui::pos2(250.0, 10.0));

            let window = if self.expanded {
                let avail = ctx.available_rect();
                base_window
                    .fixed_pos(avail.min)
                    .fixed_size(avail.size())
                    .resizable(false)
            } else {
                base_window.resizable(true)
            };

            window.show(ctx, |ui| {
                egui::ScrollArea::both().drag_to_scroll(false).show(ui, |ui| {
                    let img = egui::Image::new(egui::load::SizedTexture::new(
                        tex_id,
                        [self.canvas_width as f32, self.canvas_height as f32],
                    ))
                    .sense(egui::Sense::drag());

                    let resp = ui.add(img);

                    // Left-button drag on the image → rotate.
                    // Sense::drag() makes the Image widget claim the drag before the
                    // ScrollArea can see it; the scrollbar thumbs are separate widgets
                    // and still respond normally.
                    if resp.dragged_by(egui::PointerButton::Primary) {
                        self.viewport_drag_left = resp.drag_delta();
                    }

                    // Middle-button drag → pan
                    if resp.hovered() || resp.dragged() {
                        ui.input(|i| {
                            if i.pointer.button_down(egui::PointerButton::Middle) {
                                self.viewport_drag_middle = i.pointer.delta();
                            }
                            // Scroll wheel → zoom
                            let scroll = i.smooth_scroll_delta.y;
                            if scroll != 0.0 {
                                self.viewport_scroll = scroll;
                            }
                        });
                    }
                });
            });
        }

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
                            ui.set_min_width(160.0);
                            match metrics {
                                Some((ly1, dky)) if !ly1.is_nan() => {
                                    egui::Grid::new("metrics_grid")
                                        .num_columns(3)
                                        .spacing([4.0, 2.0])
                                        .show(ui, |ui| {
                                            let w = egui::Color32::WHITE;
                                            let m = |s: &str| egui::RichText::new(s).monospace().color(w);
                                            ui.label(m("λ₁"));
                                            ui.label(m("="));
                                            ui.label(m(&format!("{:.4}", ly1)));
                                            ui.end_row();
                                            ui.label(m("D_KY"));
                                            ui.label(m("="));
                                            ui.label(m(&format!("{:.4}", dky)));
                                            ui.end_row();
                                        });
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
                            // iter count always shown
                            {
                                let w = egui::Color32::WHITE;
                                let m = |s: &str| egui::RichText::new(s).monospace().color(w);
                                egui::Grid::new("iter_grid")
                                    .num_columns(3)
                                    .spacing([4.0, 2.0])
                                    .show(ui, |ui| {
                                        ui.label(m("iter"));
                                        ui.label(m("="));
                                        ui.label(m(&fmt_iters(self.iter_count)));
                                        ui.end_row();
                                    });
                            }
                        });
                });
        }
    }

    fn show_movie_dialog(&mut self, ctx: &Context) {
        if !self.movie_dialog_open {
            return;
        }
        let mut open = self.movie_dialog_open;
        let job_active = self.movie_job_active;
        egui::Window::new("Render Movie")
            .open(&mut open)
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                if job_active {
                    self.show_movie_progress(ui);
                } else {
                    self.show_movie_setup(ui);
                }
            });
        self.movie_dialog_open = open;
    }

    fn show_movie_setup(&mut self, ui: &mut egui::Ui) {
        ui.label("Keyframes (in order):");
        let mut remove_idx = None;
        for (i, path) in self.movie_keyframe_paths.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("{}.", i + 1));
                let name = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.display().to_string());
                ui.label(name);
                if ui.small_button("✕").clicked() {
                    remove_idx = Some(i);
                }
            });
        }
        if let Some(i) = remove_idx {
            self.movie_keyframe_paths.remove(i);
        }
        if ui.button("Add Keyframe(s)…").clicked() {
            if let Some(paths) = rfd::FileDialog::new()
                .set_title("Add Keyframe State Files")
                .add_filter("Attractor State", &["json"])
                .pick_files()
            {
                self.movie_keyframe_paths.extend(paths);
            }
        }

        ui.separator();
        ui.add(egui::DragValue::new(&mut self.movie_frames_per_step).range(1..=600).prefix("Frames/step: "));
        ui.add(egui::DragValue::new(&mut self.movie_iters_per_frame).range(1..=2_000_000_000u64).prefix("Iterations/frame: "));
        ui.checkbox(&mut self.movie_loop_back, "Loop back to first keyframe");

        ui.separator();
        let prev_output_kind = self.movie_output_kind;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.movie_output_kind, OutputKind::PngSequence, "PNG Sequence");
            ui.selectable_value(&mut self.movie_output_kind, OutputKind::Gif, "GIF");
            ui.selectable_value(&mut self.movie_output_kind, OutputKind::Mp4, "MP4");
        });
        if self.movie_output_kind != prev_output_kind {
            // The previously chosen path's extension/kind (folder vs. file) no
            // longer matches — force re-picking rather than silently reusing a
            // stale path with the wrong extension.
            self.movie_output_path = None;
        }
        if self.movie_output_kind != OutputKind::PngSequence {
            ui.add(egui::DragValue::new(&mut self.movie_fps).range(1..=120).prefix("FPS: "));
        }
        if self.movie_output_kind == OutputKind::Mp4 {
            ui.add(
                egui::Slider::new(&mut self.movie_mp4_crf, 0..=51)
                    .text("Quality (CRF, lower = better)")
                    .clamping(egui::SliderClamping::Always),
            );
        }

        ui.horizontal(|ui| {
            ui.label("Output:");
            let label = self.movie_output_path.as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(none)".to_string());
            ui.label(label);
            if ui.button("Choose…").clicked() {
                self.movie_output_path = match self.movie_output_kind {
                    OutputKind::PngSequence => rfd::FileDialog::new()
                        .set_title("Choose Output Folder")
                        .pick_folder(),
                    OutputKind::Gif => rfd::FileDialog::new()
                        .set_title("Save GIF")
                        .set_file_name("movie.gif")
                        .add_filter("GIF", &["gif"])
                        .save_file(),
                    OutputKind::Mp4 => rfd::FileDialog::new()
                        .set_title("Save MP4")
                        .set_file_name("movie.mp4")
                        .add_filter("MP4", &["mp4"])
                        .save_file(),
                };
            }
        });

        if let Some(MovieStatus::Error(msg)) = &self.movie_status_for_ui {
            ui.colored_label(egui::Color32::from_rgb(255, 100, 100), msg);
        }

        let enough_keyframes = self.movie_keyframe_paths.len() >= 2;
        let has_output = self.movie_output_path.is_some();
        if ui.add_enabled(enough_keyframes && has_output, egui::Button::new("Render")).clicked() {
            self.movie_render_requested = true;
        }
    }

    fn show_movie_progress(&mut self, ui: &mut egui::Ui) {
        match self.movie_status_for_ui.clone() {
            Some(MovieStatus::Rendering { frame_index, total_frames }) => {
                let frac = if total_frames > 0 { frame_index as f32 / total_frames as f32 } else { 0.0 };
                ui.add(egui::ProgressBar::new(frac).text(format!("{frame_index}/{total_frames}")));
                if ui.button("Cancel").clicked() {
                    self.movie_cancel_requested = true;
                }
            }
            Some(MovieStatus::Encoding) => {
                ui.label("Encoding…");
            }
            Some(MovieStatus::Done(path)) => {
                ui.label(format!("Done: {}", path.display()));
                if ui.button("Close").clicked() {
                    self.movie_close_requested = true;
                }
            }
            Some(MovieStatus::Error(msg)) => {
                ui.colored_label(egui::Color32::from_rgb(255, 100, 100), &msg);
                if ui.button("Close").clicked() {
                    self.movie_close_requested = true;
                }
            }
            Some(MovieStatus::Cancelled) => {
                ui.label("Cancelled.");
                if ui.button("Close").clicked() {
                    self.movie_close_requested = true;
                }
            }
            None => {
                ui.label("Starting…");
            }
        }
    }
}

fn fmt_iters(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
