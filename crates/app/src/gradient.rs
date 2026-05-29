/// A single color stop in a gradient.  `pos` is in [0.0, 1.0].
#[derive(Clone, Debug, PartialEq)]
pub struct ColorStop {
    pub pos: f32,
    pub rgb: [u8; 3],
}

/// A multi-stop linear gradient.
///
/// Invariants: `stops` is non-empty, sorted by `pos`, first stop always at 0.0,
/// last stop always at 1.0.  These two endpoints are "fixed" — the editor
/// does not allow the user to move or delete them.
#[derive(Clone, Debug)]
pub struct Gradient {
    pub stops: Vec<ColorStop>,
}

impl Gradient {
    /// Default density gradient: black → blue → cyan → white.
    pub fn density_default() -> Self {
        Self {
            stops: vec![
                ColorStop { pos: 0.0, rgb: [0,   0,   0  ] },
                ColorStop { pos: 0.4, rgb: [10,  20,  100] },
                ColorStop { pos: 0.7, rgb: [20,  120, 220] },
                ColorStop { pos: 1.0, rgb: [255, 255, 255] },
            ],
        }
    }

    /// Default speed gradient: black → deep red → yellow.
    pub fn speed_default() -> Self {
        Self {
            stops: vec![
                ColorStop { pos: 0.0, rgb: [0,   0,   0  ] },
                ColorStop { pos: 0.5, rgb: [180, 40,  0  ] },
                ColorStop { pos: 1.0, rgb: [255, 220, 0  ] },
            ],
        }
    }

    /// Linearly interpolate between stops to sample the color at `t ∈ [0, 1]`.
    pub fn sample(&self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let hi = self.stops.partition_point(|s| s.pos < t);
        if hi == 0 { return self.stops[0].rgb; }
        if hi >= self.stops.len() { return self.stops[self.stops.len() - 1].rgb; }
        let lo = &self.stops[hi - 1];
        let hi = &self.stops[hi];
        let span = hi.pos - lo.pos;
        let f = if span < 1e-6 { 0.0 } else { (t - lo.pos) / span };
        [
            lerp_u8(lo.rgb[0], hi.rgb[0], f),
            lerp_u8(lo.rgb[1], hi.rgb[1], f),
            lerp_u8(lo.rgb[2], hi.rgb[2], f),
        ]
    }

    /// Rasterize to a 256-texel RGBA8 row (width=256, height=1).
    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256 * 4);
        for i in 0..256usize {
            let t = i as f32 / 255.0;
            let [r, g, b] = self.sample(t);
            buf.extend_from_slice(&[r, g, b, 255]);
        }
        buf
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Gradient editor egui widget
// ---------------------------------------------------------------------------

/// Draw a gradient editor inside `ui`.  Returns `true` if the gradient changed.
///
/// - Click on empty rail area to add a stop (interpolated color at that position).
/// - Drag a movable stop to reposition it (clamped between its neighbours).
/// - Click a stop to select it; a color picker and delete button appear below.
/// - Endpoint stops (at 0.0 and 1.0) are immovable and cannot be deleted.
///
/// `selected` tracks which stop index is focused.  Pass a persistent `egui::Id`
/// unique to this editor to avoid interaction conflicts when two editors coexist.
pub fn gradient_editor(
    ui: &mut egui::Ui,
    gradient: &mut Gradient,
    selected: &mut Option<usize>,
    id: egui::Id,
) -> bool {
    let mut changed = false;
    let width = ui.available_width().max(80.0);

    // ── gradient preview bar ────────────────────────────────────────────────
    let bar_size = egui::vec2(width, 20.0);
    let (bar_rect, _) = ui.allocate_exact_size(bar_size, egui::Sense::hover());
    {
        let painter = ui.painter();
        const STRIPS: usize = 128;
        for i in 0..STRIPS {
            let t0 = i as f32 / STRIPS as f32;
            let t1 = (i + 1) as f32 / STRIPS as f32;
            let [r, g, b] = gradient.sample((t0 + t1) * 0.5);
            let x0 = bar_rect.left() + t0 * bar_rect.width();
            let x1 = (bar_rect.left() + t1 * bar_rect.width()).min(bar_rect.right());
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x0, bar_rect.top()),
                    egui::pos2(x1, bar_rect.bottom()),
                ),
                0.0,
                egui::Color32::from_rgb(r, g, b),
            );
        }
        painter.rect_stroke(
            bar_rect,
            0.0,
            egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        );
    }

    // ── stop rail ───────────────────────────────────────────────────────────
    let rail_size = egui::vec2(width, 22.0);
    let (rail_rect, rail_resp) = ui.allocate_exact_size(rail_size, egui::Sense::click());
    let rail_y = rail_rect.center().y;

    {
        let painter = ui.painter();
        painter.line_segment(
            [egui::pos2(rail_rect.left(), rail_y), egui::pos2(rail_rect.right(), rail_y)],
            egui::Stroke::new(1.5, egui::Color32::from_gray(100)),
        );
    }

    // Click on empty rail area → add a stop
    if rail_resp.clicked() {
        if let Some(ptr) = rail_resp.interact_pointer_pos() {
            let t = ((ptr.x - rail_rect.left()) / rail_rect.width()).clamp(0.01, 0.99);
            let too_close = gradient.stops.iter().any(|s| (s.pos - t).abs() < 0.03);
            if !too_close {
                let rgb = gradient.sample(t);
                let idx = gradient.stops.partition_point(|s| s.pos < t);
                gradient.stops.insert(idx, ColorStop { pos: t, rgb });
                // Adjust selected index if it was after the insertion point.
                *selected = Some(idx);
                changed = true;
            }
        }
    }

    // Draw and interact with each stop handle
    let stop_r = 7.0;
    let n = gradient.stops.len();

    // Collect draw calls after all interactions (avoids painter borrow during interact calls)
    let mut draw_calls: Vec<(egui::Pos2, f32, egui::Color32, egui::Stroke, bool)> = Vec::new();

    for i in 0..n {
        let is_endpoint = i == 0 || i == n - 1;
        let cx = rail_rect.left() + gradient.stops[i].pos * rail_rect.width();
        let center = egui::pos2(cx, rail_y);

        let handle_rect = egui::Rect::from_center_size(
            center,
            egui::vec2(stop_r * 2.4, stop_r * 2.4),
        );
        let handle_resp = ui.interact(handle_rect, id.with(i), egui::Sense::click_and_drag());

        if handle_resp.clicked() {
            *selected = Some(i);
        }

        if !is_endpoint && handle_resp.dragged() {
            let dx = handle_resp.drag_delta().x / rail_rect.width();
            let lo_bound = if i > 1 { gradient.stops[i - 1].pos + 0.005 } else { 0.005 };
            let hi_bound = if i < n - 2 { gradient.stops[i + 1].pos - 0.005 } else { 0.995 };
            gradient.stops[i].pos = (gradient.stops[i].pos + dx).clamp(lo_bound, hi_bound);
            changed = true;
        }

        let is_selected = *selected == Some(i);
        let [r, g, b] = gradient.stops[i].rgb;
        let fill = egui::Color32::from_rgb(r, g, b);
        let stroke = egui::Stroke::new(
            if is_selected { 2.0 } else { 1.5 },
            if is_selected { egui::Color32::WHITE } else { egui::Color32::from_gray(160) },
        );
        // Recompute cx after possible drag
        let cx2 = rail_rect.left() + gradient.stops[i].pos * rail_rect.width();
        draw_calls.push((egui::pos2(cx2, rail_y), stop_r, fill, stroke, is_endpoint));
    }

    // Paint stops
    {
        let painter = ui.painter();
        for (center, r, fill, stroke, is_endpoint) in &draw_calls {
            if *is_endpoint {
                let half = *r;
                let rect = egui::Rect::from_center_size(*center, egui::vec2(half * 2.0, half * 2.0));
                painter.rect_filled(rect, 2.0, *fill);
                painter.rect_stroke(rect, 2.0, *stroke);
            } else {
                painter.circle_filled(*center, *r, *fill);
                painter.circle_stroke(*center, *r, *stroke);
            }
        }
    }

    // ── selected-stop controls ───────────────────────────────────────────────
    if let Some(idx) = *selected {
        if idx < gradient.stops.len() {
            let is_endpoint = idx == 0 || idx == gradient.stops.len() - 1;

            // Extract color as f32 first (avoids borrow conflict with remove below)
            let mut rgb_f = {
                let stop = &gradient.stops[idx];
                [
                    stop.rgb[0] as f32 / 255.0,
                    stop.rgb[1] as f32 / 255.0,
                    stop.rgb[2] as f32 / 255.0,
                ]
            };

            let mut color_changed = false;
            let mut delete_pressed = false;

            ui.horizontal(|ui| {
                if egui::color_picker::color_edit_button_rgb(ui, &mut rgb_f).changed() {
                    color_changed = true;
                }
                if !is_endpoint && ui.small_button("✕ Remove").clicked() {
                    delete_pressed = true;
                }
            });

            if color_changed {
                gradient.stops[idx].rgb = [
                    (rgb_f[0] * 255.0).round() as u8,
                    (rgb_f[1] * 255.0).round() as u8,
                    (rgb_f[2] * 255.0).round() as u8,
                ];
                changed = true;
            }
            if delete_pressed {
                gradient.stops.remove(idx);
                *selected = None;
                changed = true;
            }
        }
    }

    changed
}
