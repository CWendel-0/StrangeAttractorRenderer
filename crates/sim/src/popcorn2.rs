use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b]
// Equations (discrete map, z constant for 2-D attractor) — distinct from
// Popcorn: this variant subtracts the perturbation from the previous
// coordinate instead of replacing it outright.
//   x' = x - a*sin(y + tan(b*y))
//   y' = y - a*sin(x' + tan(b*x'))
//   z' = 0
//
// Source: Popcorn2.cof — a=0.05, b=3.0, start (0.60, 0.20).
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -0.5, max: 0.5, default: 0.05 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: -6.0, max: 6.0, default: 3.0  },
];

pub struct Popcorn2 {
    state:  Vec3,
    params: [f32; 2],
}

impl Popcorn2 {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.60, 0.20, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.60, 0.20, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b) = (self.params[0], self.params[1]);
        let (x, y) = (self.state.x, self.state.y);
        let nx = x - a * (y + (b * y).tan()).sin();
        let ny = y - a * (nx + (b * nx).tan()).sin();
        self.state.x = nx;
        self.state.y = ny;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for Popcorn2 {
    fn default() -> Self { Self::new() }
}

impl Attractor for Popcorn2 {
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
        self.state = Vec3::new(0.60, 0.20, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
