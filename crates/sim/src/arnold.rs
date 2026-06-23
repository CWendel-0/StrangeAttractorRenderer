use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a]
// Equations (discrete map on the unit torus, x' kept from previous x via xa):
//   x' = (xa + y + a*cos(2*PI*y)) mod 1
//   y' = (xa + 2*y) mod 1
// (xa is the *old* x value, i.e. the update uses the previous x for both lines)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: 0.15 },
];

pub struct Arnold {
    state:  Vec3,
    params: [f32; 1],
}

impl Arnold {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.0),
            params: [DESCS[0].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.1, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let a = self.params[0];
        let xa = self.state.x;
        let y = self.state.y;
        let nx = (xa + y + a * (2.0 * std::f32::consts::PI * y).cos()).rem_euclid(1.0);
        let ny = (xa + 2.0 * y).rem_euclid(1.0);
        self.state.x = nx;
        self.state.y = ny;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for Arnold {
    fn default() -> Self { Self::new() }
}

impl Attractor for Arnold {
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
        self.state = Vec3::new(0.1, 0.1, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
