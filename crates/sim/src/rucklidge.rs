use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [k, a, dt]
// Equations (Euler form):
//   x' = x + dt*(-k*x + a*y - y*z)
//   y' = y + dt*x
//   z' = z + dt*(-z + y²)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "k",  kind: ParamKind::Continuous, min: 0.0, max: 6.0,  default: 2.0 },
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 20.1, default: 6.7 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.05, default: 0.01 },
];

pub struct Rucklidge {
    state:  Vec3,
    params: [f32; 3],
}

impl Rucklidge {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, 0.0, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, 0.0, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (k, a, dt) = (self.params[0], self.params[1], self.params[2]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (-k * x + a * y - y * z);
        self.state.y = y + dt * x;
        self.state.z = z + dt * (-z + y * y);
    }
}

impl Default for Rucklidge {
    fn default() -> Self { Self::new() }
}

impl Attractor for Rucklidge {
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
        self.state = Vec3::new(1.0, 0.0, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
