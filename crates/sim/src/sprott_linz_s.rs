use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Sprott-Linz S
// Params: [a, dt]
// Equations (Euler form):
//   x' = x + dt*(-x - a*y)
//   y' = y + dt*(x + z²)
//   z' = z + dt*(1 + x)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 12.0,  default: 4.0 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.035, default: 0.007 },
];

pub struct SprottLinzS {
    state:  Vec3,
    params: [f32; 2],
}

impl SprottLinzS {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.1),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2900 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.1, 0.1);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, dt) = (self.params[0], self.params[1]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (-x - a * y);
        self.state.y = y + dt * (x + z * z);
        self.state.z = z + dt * (1.0 + x);
    }
}

impl Default for SprottLinzS {
    fn default() -> Self { Self::new() }
}

impl Attractor for SprottLinzS {
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
        self.state = Vec3::new(0.1, 0.1, 0.1);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
