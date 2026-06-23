use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, dt]
// Equations (Euler form):
//   x' = x + dt*(a*x - y*z)
//   y' = y + dt*(b*y + x*z)
//   z' = z + dt*(c*z + x*y/3)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0,  max: 15.0,  default: 5.0  },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: -25.0, max: 0.0,  default: -10.0 },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: -0.95, max: 0.0,  default: -0.38 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,  max: 0.02,  default: 0.004 },
];

pub struct ChenLee {
    state:  Vec3,
    params: [f32; 4],
}

impl ChenLee {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, 0.0, 4.50),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..5000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, 0.0, 4.50);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, c, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (a * x - y * z);
        self.state.y = y + dt * (b * y + x * z);
        self.state.z = z + dt * (c * z + x * y / 3.0);
    }
}

impl Default for ChenLee {
    fn default() -> Self { Self::new() }
}

impl Attractor for ChenLee {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[3].max(1e-9);
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(1.0, 0.0, 4.50);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
