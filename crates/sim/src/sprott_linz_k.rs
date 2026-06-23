use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Sprott-Linz K
// Params: [a, dt]
// Equations (Euler form):
//   x' = x + dt*(x*y - z)
//   y' = y + dt*(x - y)
//   z' = z + dt*(x + a*z)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 0.9, default: 0.3 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.1, default: 0.02 },
];

pub struct SprottLinzK {
    state:  Vec3,
    params: [f32; 2],
}

impl SprottLinzK {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.0, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..1000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.0, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, dt) = (self.params[0], self.params[1]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (x * y - z);
        self.state.y = y + dt * (x - y);
        self.state.z = z + dt * (x + a * z);
    }
}

impl Default for SprottLinzK {
    fn default() -> Self { Self::new() }
}

impl Attractor for SprottLinzK {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[1].max(1e-9);
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
