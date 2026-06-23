use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c]
// Equations (discrete map, z constant for 2-D attractor):
//   x' = y - sign(x)*sqrt(|b*x - c|)
//   y' = a - x
//   z' = 0
//
// Source: Hopalong.cof — a=0.4, b=1.0, c=0.0, start (0.50, 0.50).
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -2.0, max: 2.0, default: 0.4 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: -2.0, max: 2.0, default: 1.0 },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: -2.0, max: 2.0, default: 0.0 },
];

fn sign(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else if x == 0.0 { 0.0 } else { -1.0 }
}

pub struct Hopalong {
    state:  Vec3,
    params: [f32; 3],
}

impl Hopalong {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.50, 0.50, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
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
        let (a, b, c) = (self.params[0], self.params[1], self.params[2]);
        let (x, y) = (self.state.x, self.state.y);
        let nx = y - sign(x) * (b * x - c).abs().sqrt();
        let ny = a - x;
        self.state.x = nx;
        self.state.y = ny;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for Hopalong {
    fn default() -> Self { Self::new() }
}

impl Attractor for Hopalong {
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
