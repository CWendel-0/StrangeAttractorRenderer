use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, c, d, e, f, dT]
// Equations (Euler form):
//   x' = x + dt*(a*(y-x) + d*x*z)
//   y' = y + dt*(f*y - x*z)
//   z' = z + dt*(c*z + x*y - e*x²)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 120.0, default: 40.0  },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: 0.0, max: 2.5,   default: 0.833 },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0, max: 1.5,   default: 0.5   },
    ParamDesc { name: "e",  kind: ParamKind::Continuous, min: 0.0, max: 1.95,  default: 0.65  },
    ParamDesc { name: "f",  kind: ParamKind::Continuous, min: 0.0, max: 60.0,  default: 20.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.003, default: 0.001 },
];

pub struct Tsucs1 {
    state:  Vec3,
    params: [f32; 6],
}

impl Tsucs1 {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 1.0, -0.1),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default,
                DESCS[3].default, DESCS[4].default, DESCS[5].default,
            ],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..5000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 1.0, -0.1);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, c, d, e, f, dt) = (
            self.params[0], self.params[1], self.params[2],
            self.params[3], self.params[4], self.params[5],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let dx = a * (y - x) + d * x * z;
        let dy = f * y - x * z;
        let dz = c * z + x * y - e * x * x;
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for Tsucs1 {
    fn default() -> Self { Self::new() }
}

impl Attractor for Tsucs1 {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[5].max(1e-9);
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(0.1, 1.0, -0.1);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
