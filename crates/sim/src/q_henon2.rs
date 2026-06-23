use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};
use std::f32::consts::PI;

// Params: [a, b, c, d]
// Quasiperiodic Henon map, variant 2 (3-variable skew-product map; w is a
// driven phase that feeds back into x, so all three components are needed
// to advance the dynamics):
//   x' = a - x^2 + b*y + c*cos(2*PI*w)
//   y' = x
//   w' = (w + d) mod 1
// Original initial values: x0 = 0.1, y0 = 0.1, w0 = 0.1.

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0, max: 2.0, default: 1.09 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: 0.0, max: 1.0, default: 0.3 },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: 0.0, max: 1.0, default: 0.2 },
    ParamDesc { name: "d", kind: ParamKind::Continuous, min: 0.0, max: 1.0, default: 0.618 },
];

pub struct QHenon2 {
    state:  Vec3, // x, y, w
    params: [f32; 4],
}

impl QHenon2 {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.1),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.1, 0.1);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b, c, d) = (self.params[0], self.params[1], self.params[2], self.params[3]);
        let (x, y, w) = (self.state.x, self.state.y, self.state.z);
        let nx = a - x * x + b * y + c * (2.0 * PI * w).cos();
        let ny = x;
        let nw = (w + d).rem_euclid(1.0);
        self.state = Vec3::new(nx, ny, nw);
    }
}

impl Default for QHenon2 {
    fn default() -> Self { Self::new() }
}

impl Attractor for QHenon2 {
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
        self.state = Vec3::new(0.1, 0.1, 0.1);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
