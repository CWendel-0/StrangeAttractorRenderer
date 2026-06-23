use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};
use std::f32::consts::PI;

// Params: [a, b, c]
// Equations (discrete 2-D map, logistic map coupled to a circle map):
//   x' = a*x*(1-x) + b*sin(2*PI*y)
//   y' = (y + c) mod 1
//   z' = 0

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0, max: 4.0, default: 3.0 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: 0.0, max: 1.0, default: 0.157 },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: 0.0, max: 1.0, default: 0.618 },
];

pub struct NishikawaKaneko {
    state:  Vec3, // x, y, z=0
    params: [f32; 3],
}

impl NishikawaKaneko {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
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
        let (a, b, c) = (self.params[0], self.params[1], self.params[2]);
        let (x, y) = (self.state.x, self.state.y);
        let nx = a * x * (1.0 - x) + b * (2.0 * PI * y).sin();
        let ny = (y + c).rem_euclid(1.0);
        self.state.x = nx;
        self.state.y = ny;
    }
}

impl Default for NishikawaKaneko {
    fn default() -> Self { Self::new() }
}

impl Attractor for NishikawaKaneko {
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
