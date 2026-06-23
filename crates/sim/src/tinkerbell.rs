use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Tinkerbell Map (Juergen Meier, 3d-meier.de, "Tinkerbell.cof")
//
// Params: [a, b, c, d]
// Equations (discrete map, z constant for 2-D attractor):
//   x' = x^2 - y^2 + a*x + b*y
//   y' = 2*x*y + c*x + d*y
//   z' = 0

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -3.0, max: 3.0, default: 0.90 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: -3.0, max: 3.0, default: -0.601 },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: -3.0, max: 3.0, default: 2.00 },
    ParamDesc { name: "d", kind: ParamKind::Continuous, min: -3.0, max: 3.0, default: 0.50 },
];

pub struct Tinkerbell {
    state:  Vec3,
    params: [f32; 4],
}

impl Tinkerbell {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(-0.72, -0.64, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(-0.72, -0.64, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b, c, d) = (self.params[0], self.params[1], self.params[2], self.params[3]);
        let (x, y) = (self.state.x, self.state.y);
        self.state.x = x * x - y * y + a * x + b * y;
        self.state.y = 2.0 * x * y + c * x + d * y;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for Tinkerbell {
    fn default() -> Self { Self::new() }
}

impl Attractor for Tinkerbell {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.map_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length();
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(-0.72, -0.64, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
