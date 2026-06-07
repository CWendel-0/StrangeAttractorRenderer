// Horizontal slider with a velocity-driven fine mode.
//
// Clicking far from the thumb → direct position mapping (coarse).
// Clicking within FINE_ZONE_PX of the thumb → velocity mode (fine).
// Speed scales continuously: slower the closer to the thumb centre,
// so the thumb acts as a large soft target for precision edits.

use egui::{Color32, Id, Rect, Sense, Stroke, Ui, Vec2, pos2};

const FINE_ZONE_PX: f32 = 24.0;
const TRACK_H:      f32 = 4.0;
const THUMB_R:      f32 = 7.0;
const LABEL_W:      f32 = 90.0;

#[derive(Clone)]
struct DragState {
    fine:         bool,
    speed_factor: f32, // multiplier on (range / track_w) per cursor pixel
}

impl Default for DragState {
    fn default() -> Self {
        Self { fine: false, speed_factor: 1.0 }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Draw a velocity-driven slider. Returns `true` if the value changed this frame.
pub fn velocity_slider(
    ui:    &mut Ui,
    value: &mut f32,
    min:   f32,
    max:   f32,
    label: &str,
    id:    Id,
) -> bool {
    let range   = max - min;
    let height  = ui.spacing().interact_size.y;
    let avail_w = ui.available_width();
    let track_w = (avail_w - LABEL_W - THUMB_R * 2.0).max(30.0);

    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(avail_w, height),
        Sense::click_and_drag(),
    );

    let track_left  = rect.left() + THUMB_R;
    let track_right = track_left + track_w;
    let cy          = rect.center().y;
    let track_rect  = Rect::from_min_max(
        pos2(track_left,  cy - TRACK_H * 0.5),
        pos2(track_right, cy + TRACK_H * 0.5),
    );

    // Thumb x before any update this frame (used to classify click).
    let thumb_x0 = lerp(track_left, track_right,
                        ((*value - min) / range).clamp(0.0, 1.0));

    // ---- Interaction -------------------------------------------------------
    let prev = *value;

    if response.drag_started() {
        if let Some(ptr) = response.interact_pointer_pos() {
            let dist = (ptr.x - thumb_x0).abs();
            let fine = dist <= FINE_ZONE_PX;
            // speed_factor: edge of zone → 10 % of coarse; dead centre → 0.5 %
            let speed_factor = if fine {
                let t = 1.0 - dist / FINE_ZONE_PX;
                lerp(0.10, 0.005, t * t)
            } else {
                1.0
            };
            ui.memory_mut(|m| m.data.insert_temp(id, DragState { fine, speed_factor }));
            if !fine {
                // Coarse: jump immediately to the clicked position.
                let nt = ((ptr.x - track_left) / track_w).clamp(0.0, 1.0);
                *value = min + nt * range;
            }
        }
    } else if response.dragged() {
        let st: DragState = ui.memory(|m| m.data.get_temp(id).unwrap_or_default());
        if st.fine {
            // Fine: accumulate scaled delta.
            let dx = response.drag_delta().x;
            *value = (*value + dx * (range / track_w) * st.speed_factor).clamp(min, max);
        } else {
            // Coarse: follow cursor position directly.
            if let Some(ptr) = response.interact_pointer_pos() {
                let nt = ((ptr.x - track_left) / track_w).clamp(0.0, 1.0);
                *value = min + nt * range;
            }
        }
    }

    // ---- Painting ----------------------------------------------------------
    if ui.is_rect_visible(rect) {
        let is_active = response.dragged() || response.hovered();
        let vis = ui.visuals();

        let thumb_x = lerp(track_left, track_right,
                           ((*value - min) / range).clamp(0.0, 1.0));
        let thumb_c = pos2(thumb_x, cy);

        // Track background
        ui.painter().rect_filled(track_rect, TRACK_H * 0.5,
                                 vis.widgets.inactive.bg_fill);

        // Fill (left of thumb)
        let fill = Rect::from_min_max(
            track_rect.min,
            pos2(thumb_x.max(track_left), track_rect.max.y),
        );
        ui.painter().rect_filled(fill, TRACK_H * 0.5,
                                 Color32::from_rgb(60, 120, 220));

        // Fine-zone ring — faint when hovered, brighter when actively dragging fine
        if is_active {
            let st: DragState = ui.memory(|m| m.data.get_temp(id).unwrap_or_default());
            let alpha = if response.dragged() && st.fine { 110u8 } else { 35u8 };
            ui.painter().circle_stroke(
                thumb_c,
                FINE_ZONE_PX,
                Stroke::new(1.0,
                    Color32::from_rgba_premultiplied(180, 180, 255, alpha)),
            );
        }

        // Thumb
        let thumb_r = if response.dragged() { THUMB_R + 1.5 } else { THUMB_R };
        let thumb_col = if is_active {
            vis.widgets.hovered.bg_fill
        } else {
            vis.widgets.inactive.fg_stroke.color
        };
        ui.painter().circle_filled(thumb_c, thumb_r, thumb_col);

        // "value  label" text to the right of the track
        let text = format!("{:.3}  {label}", *value);
        ui.painter().text(
            pos2(track_right + 8.0, cy),
            egui::Align2::LEFT_CENTER,
            text,
            egui::FontId::default(),
            vis.text_color(),
        );
    }

    *value != prev
}
