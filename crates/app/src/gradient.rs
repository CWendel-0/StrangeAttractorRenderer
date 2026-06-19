/// Color space used when interpolating between stops.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InterpMode {
    Srgb,
    Oklab,
    Oklch,
}

/// A single color stop in a gradient.  `pos` is in [0.0, 1.0].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ColorStop {
    pub pos: f32,
    pub rgb: [u8; 3],
}

/// A multi-stop linear gradient.
///
/// Invariants: `stops` is non-empty, sorted by `pos`, first stop always at 0.0,
/// last stop always at 1.0.  These two endpoints are "fixed" — the editor
/// does not allow the user to move or delete them.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Gradient {
    pub stops: Vec<ColorStop>,
    pub interp_mode: InterpMode,
}

// ---------------------------------------------------------------------------
// Oklab conversions (Björn Ottosson, 2020)
// ---------------------------------------------------------------------------

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    [
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ]
}

fn oklab_to_linear_rgb(ll: f32, a: f32, b: f32) -> [f32; 3] {
    let l_ = ll + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = ll - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = ll - 0.0894841775 * a - 1.2914855480 * b;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    [
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]
}

fn rgb_u8_to_oklab(rgb: [u8; 3]) -> [f32; 3] {
    let r = srgb_to_linear(rgb[0] as f32 / 255.0);
    let g = srgb_to_linear(rgb[1] as f32 / 255.0);
    let b = srgb_to_linear(rgb[2] as f32 / 255.0);
    linear_rgb_to_oklab(r, g, b)
}

fn oklab_to_rgb_u8(lab: [f32; 3]) -> [u8; 3] {
    let [r, g, b] = oklab_to_linear_rgb(lab[0], lab[1], lab[2]);
    [
        (linear_to_srgb(r.clamp(0.0, 1.0)) * 255.0).round() as u8,
        (linear_to_srgb(g.clamp(0.0, 1.0)) * 255.0).round() as u8,
        (linear_to_srgb(b.clamp(0.0, 1.0)) * 255.0).round() as u8,
    ]
}

// ---------------------------------------------------------------------------
// Minimal xorshift32 PRNG — only used for gradient randomization
// ---------------------------------------------------------------------------

fn rng_next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn rng_f32(state: &mut u32) -> f32 {
    rng_next(state) as f32 / u32::MAX as f32
}

fn oklab_to_oklch(lab: [f32; 3]) -> [f32; 3] {
    let c = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
    let h = lab[2].atan2(lab[1]);
    [lab[0], c, h]
}

fn oklch_to_oklab(lch: [f32; 3]) -> [f32; 3] {
    [lch[0], lch[1] * lch[2].cos(), lch[1] * lch[2].sin()]
}

fn lerp_hue(a: f32, b: f32, t: f32) -> f32 {
    use std::f32::consts::TAU;
    let diff = (b - a).rem_euclid(TAU);
    let delta = if diff > std::f32::consts::PI { diff - TAU } else { diff };
    a + delta * t
}

impl Gradient {
    /// Default density gradient: yellow → deep red → blue → cyan.
    pub fn density_default() -> Self {
        Self {
            stops: vec![
                ColorStop { pos: 0.00, rgb: [210, 220,   0] },
                ColorStop { pos: 0.35, rgb: [170,  15,  40] },
                ColorStop { pos: 0.70, rgb: [ 40,  70, 200] },
                ColorStop { pos: 1.00, rgb: [ 70, 210, 230] },
            ],
            interp_mode: InterpMode::Oklab,
        }
    }

    /// Default speed gradient: cyan → purple.
    pub fn speed_default() -> Self {
        Self {
            stops: vec![
                ColorStop { pos: 0.0, rgb: [  0, 185, 220] },
                ColorStop { pos: 1.0, rgb: [110,   0, 210] },
            ],
            interp_mode: InterpMode::Oklab,
        }
    }

    /// Sample the color at `t ∈ [0, 1]`, interpolating in `self.interp_mode` color space.
    pub fn sample(&self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let hi = self.stops.partition_point(|s| s.pos < t);
        if hi == 0 { return self.stops[0].rgb; }
        if hi >= self.stops.len() { return self.stops[self.stops.len() - 1].rgb; }
        let lo = &self.stops[hi - 1];
        let hi = &self.stops[hi];
        let span = hi.pos - lo.pos;
        let f = if span < 1e-6 { 0.0 } else { (t - lo.pos) / span };

        match self.interp_mode {
            InterpMode::Srgb => [
                lerp_u8(lo.rgb[0], hi.rgb[0], f),
                lerp_u8(lo.rgb[1], hi.rgb[1], f),
                lerp_u8(lo.rgb[2], hi.rgb[2], f),
            ],
            InterpMode::Oklab => {
                let a = rgb_u8_to_oklab(lo.rgb);
                let b = rgb_u8_to_oklab(hi.rgb);
                oklab_to_rgb_u8([
                    a[0] + (b[0] - a[0]) * f,
                    a[1] + (b[1] - a[1]) * f,
                    a[2] + (b[2] - a[2]) * f,
                ])
            }
            InterpMode::Oklch => {
                let a = oklab_to_oklch(rgb_u8_to_oklab(lo.rgb));
                let b = oklab_to_oklch(rgb_u8_to_oklab(hi.rgb));
                oklab_to_rgb_u8(oklch_to_oklab([
                    a[0] + (b[0] - a[0]) * f,
                    a[1] + (b[1] - a[1]) * f,
                    lerp_hue(a[2], b[2], f),
                ]))
            }
        }
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

    /// Replace stops with 2–4 randomly generated colors at random positions.
    /// Endpoints (pos 0.0 and 1.0) are always present; intermediate stops get
    /// sorted random positions. Colors are generated in OKLCh for vibrancy.
    pub fn randomize(&mut self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(0xDEAD_BEEF);
        let mut rng = seed.max(1);

        let n = 2 + (rng_next(&mut rng) % 3) as usize; // 2, 3, or 4 stops

        // Random middle positions, sorted.
        let mut mid: Vec<f32> = (0..n.saturating_sub(2))
            .map(|_| 0.05 + rng_f32(&mut rng) * 0.90)
            .collect();
        mid.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let positions: Vec<f32> = std::iter::once(0.0f32)
            .chain(mid)
            .chain(std::iter::once(1.0f32))
            .collect();

        self.stops = positions
            .iter()
            .map(|&pos| {
                let l = 0.15 + rng_f32(&mut rng) * 0.75; // lightness  [0.15, 0.90]
                let c = 0.06 + rng_f32(&mut rng) * 0.22; // chroma     [0.06, 0.28]
                let h = rng_f32(&mut rng) * std::f32::consts::TAU;
                let rgb = oklab_to_rgb_u8(oklch_to_oklab([l, c, h]));
                ColorStop { pos, rgb }
            })
            .collect();
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

    // ── interpolation mode toggle ───────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.label("Interp:");
        for (mode, label) in [(InterpMode::Srgb, "sRGB"), (InterpMode::Oklab, "Oklab"), (InterpMode::Oklch, "OKLCh")] {
            if ui.selectable_label(gradient.interp_mode == mode, label).clicked()
                && gradient.interp_mode != mode
            {
                gradient.interp_mode = mode;
                changed = true;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("↺ Randomize").clicked() {
                gradient.randomize();
                changed = true;
            }
        });
    });

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
