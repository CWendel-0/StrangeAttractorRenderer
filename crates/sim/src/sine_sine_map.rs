use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a]
// Discrete 2-D map:
//   x' = sin(xa) - sin(a*y)
//   y' = xa
// (xa is the previous x value). z stays 0; camera looks down z-axis.

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -4.0, max: 4.0, default: 2.0 },
];

pub struct SineSineMap {
    state:  Vec3, // x = state.x, y = state.y
    params: [f32; 1],
}

impl SineSineMap {
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
        let x = xa.sin() - (a * y).sin();
        self.state.x = x;
        self.state.y = xa;
    }
}

impl Default for SineSineMap {
    fn default() -> Self { Self::new() }
}

impl Attractor for SineSineMap {
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
