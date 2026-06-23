use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, dt]
// Equations (Euler form):
//   x' = x + dt*(x + y + y*z)
//   y' = y + dt*(y*z - x*z)
//   z' = z + dt*(-z - a*x*y + 1)
//
// Source: Four-Wing3.cof — a=1.0, Delta=0.001, start (1.0, -2.0, 1.0).
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: -3.0, max: 3.0,  default: 1.0   },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,  max: 0.01, default: 0.001 },
];

pub struct FourWing3 {
    state:  Vec3,
    params: [f32; 2],
}

impl FourWing3 {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, -2.0, 1.0),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, -2.0, 1.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, dt) = (self.params[0], self.params[1]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (x + y + y * z);
        self.state.y = y + dt * (y * z - x * z);
        self.state.z = z + dt * (-z - a * x * y + 1.0);
    }
}

impl Default for FourWing3 {
    fn default() -> Self { Self::new() }
}

impl Attractor for FourWing3 {
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
        self.state = Vec3::new(1.0, -2.0, 1.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
