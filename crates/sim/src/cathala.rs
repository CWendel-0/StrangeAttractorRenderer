use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b]
// Equations (discrete map, z constant at 0 for 2-D attractor):
//   x' = a*x + y
//   y' = b + x²
//   (uses the PREVIOUS x value for the y update, matching the .cof source
//    where y is computed from xa, the x value before this iteration's update)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -2.1, max: 2.1, default: 0.7  },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: -2.46, max: 2.46, default: -0.82 },
];

pub struct Cathala {
    state:  Vec3,
    params: [f32; 2],
}

impl Cathala {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.50, 0.50, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.50, 0.50, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b) = (self.params[0], self.params[1]);
        let xa = self.state.x;
        let nx = a * xa + self.state.y;
        let ny = b + xa * xa;
        self.state.x = nx;
        self.state.y = ny;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for Cathala {
    fn default() -> Self { Self::new() }
}

impl Attractor for Cathala {
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
        self.state = Vec3::new(0.50, 0.50, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
