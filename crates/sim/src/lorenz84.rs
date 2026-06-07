use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, F, G, dt]
// Equations (Euler form):
//   x' = x + dt*(-a*x - y² - z² + a*F)
//   y' = y + dt*(-y + x*y - b*x*z + G)
//   z' = z + dt*(-z + b*x*y + x*z)
//
// Canonical chaotic parameters: a=0.25, b=4, F=8, G=1 → attractor in ~[-3,3]
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.05, max: 1.0,  default: 0.25 },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0,  max: 8.0,  default: 4.0  },
    ParamDesc { name: "F",  kind: ParamKind::Continuous, min: 0.0,  max: 16.0, default: 8.0  },
    ParamDesc { name: "G",  kind: ParamKind::Continuous, min: -4.0, max: 4.0,  default: 1.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.001, max: 0.05, default: 0.01 },
];

pub struct Lorenz84 {
    state:  Vec3,
    params: [f32; 5],
}

impl Lorenz84 {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.0, 0.0),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default,
                DESCS[3].default, DESCS[4].default,
            ],
        };
        s.warm_up(); // 2000 × dt=0.01 = 20 time units, sufficient to land on attractor
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.0, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, f, g, dt) = (
            self.params[0], self.params[1], self.params[2],
            self.params[3], self.params[4],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (-a * x - y * y - z * z + a * f);
        self.state.y = y + dt * (-y + x * y - b * x * z + g);
        self.state.z = z + dt * (-z + b * x * y + x * z);
    }
}

impl Default for Lorenz84 {
    fn default() -> Self { Self::new() }
}

impl Attractor for Lorenz84 {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[4].max(1e-9);
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(0.1, 0.0, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
