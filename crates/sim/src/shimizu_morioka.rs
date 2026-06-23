use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, dt]
// Equations (Euler form):
//   x' = x + dt*y
//   y' = y + dt*((1-z)*x - a*y)
//   z' = z + dt*(x² - b*z)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 2.25, default: 0.75 },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0, max: 1.35, default: 0.45 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.05, default: 0.01 },
];

pub struct ShimizuMorioka {
    state:  Vec3,
    params: [f32; 3],
}

impl ShimizuMorioka {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.0, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
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
        let (a, b, dt) = (self.params[0], self.params[1], self.params[2]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * y;
        self.state.y = y + dt * ((1.0 - z) * x - a * y);
        self.state.z = z + dt * (x * x - b * z);
    }
}

impl Default for ShimizuMorioka {
    fn default() -> Self { Self::new() }
}

impl Attractor for ShimizuMorioka {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[2].max(1e-9);
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
