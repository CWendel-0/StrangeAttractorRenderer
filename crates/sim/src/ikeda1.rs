use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Ikeda 1 Attraktor (Juergen Meier, 3d-meier.de, "Ikeda1.cof")
// Generalized 4-parameter form of the Ikeda map (distinct from Ikeda.cof,
// which has only 2 free constants and a fixed leading "+1" offset).
//
// Params: [a, b, c, d]
// Equations (discrete map, z constant):
//   t = c - d/(1 + x² + y²)
//   x' = a + b*(x*cos(t) - y*sin(t))
//   y' = b*(x*sin(t) + y*cos(t))
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -5.0,  max: 5.0,   default: 2.16  },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: 0.0,   max: 1.0,   default: 0.65  },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: -10.0, max: 10.0,  default: 4.69  },
    ParamDesc { name: "d", kind: ParamKind::Continuous, min: 0.0,   max: 60.0,  default: 37.65 },
];

pub struct Ikeda1 {
    state:  Vec3,
    params: [f32; 4],
}

impl Ikeda1 {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.10, 0.10, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.10, 0.10, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b, c, d) = (self.params[0], self.params[1], self.params[2], self.params[3]);
        let (x, y) = (self.state.x, self.state.y);
        let t = c - d / (1.0 + x * x + y * y);
        self.state.x = a + b * (x * t.cos() - y * t.sin());
        self.state.y = b * (x * t.sin() + y * t.cos());
    }
}

impl Default for Ikeda1 {
    fn default() -> Self { Self::new() }
}

impl Attractor for Ikeda1 {
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
        self.state = Vec3::new(0.10, 0.10, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
