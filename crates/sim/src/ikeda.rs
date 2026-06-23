use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Ikeda Attraktor (Juergen Meier, 3d-meier.de, "Ikeda.cof")
//
// Params: [c, u]
// Equations (discrete map, z constant):
//   t = c - 6/(1 + x² + y²)
//   x' = 1 + u*(x*cos(t) - y*sin(t))
//   y' = u*(x*sin(t) + y*cos(t))
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: -2.0, max: 2.0, default: 0.40 },
    ParamDesc { name: "u", kind: ParamKind::Continuous, min: 0.0,  max: 1.0, default: 0.90 },
];

pub struct Ikeda {
    state:  Vec3,
    params: [f32; 2],
}

impl Ikeda {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.10, 0.10, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
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
        let (c, u) = (self.params[0], self.params[1]);
        let (x, y) = (self.state.x, self.state.y);
        let t = c - 6.0 / (1.0 + x * x + y * y);
        self.state.x = 1.0 + u * (x * t.cos() - y * t.sin());
        self.state.y = u * (x * t.sin() + y * t.cos());
    }
}

impl Default for Ikeda {
    fn default() -> Self { Self::new() }
}

impl Attractor for Ikeda {
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
