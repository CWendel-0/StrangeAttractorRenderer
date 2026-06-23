use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b]
// Equations (discrete map, z constant for 2-D attractor):
//   x' = (1 - a·x - b·y)·(a·x + b·y)
//   y' = x
//   z' = 0
//
// Source: Marotto-Lorenz.cof (Juergen Meier).

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0, max: 6.0, default: 2.5 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: 0.0, max: 6.0, default: 2.0 },
];

pub struct MarottoLorenz {
    state:  Vec3,
    params: [f32; 2],
}

impl MarottoLorenz {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
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
        let (a, b) = (self.params[0], self.params[1]);
        let (x, y) = (self.state.x, self.state.y);
        let s = a * x + b * y;
        self.state.x = (1.0 - s) * s;
        self.state.y = x;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for MarottoLorenz {
    fn default() -> Self { Self::new() }
}

impl Attractor for MarottoLorenz {
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
