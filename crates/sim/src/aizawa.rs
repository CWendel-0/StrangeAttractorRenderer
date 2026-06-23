use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d, e, f, dt]
// Equations (Euler form):
//   x' = x + dt*( (z-b)*x - d*y )
//   y' = y + dt*( d*x + (z-b)*y )
//   z' = z + dt*( c + a*z - z³/3 - (x²+y²)*(1+e*z) + f*z*x³ )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 2.85, default: 0.95 },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0, max: 2.1,  default: 0.7  },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: 0.0, max: 1.8,  default: 0.6  },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0, max: 10.5, default: 3.5  },
    ParamDesc { name: "e",  kind: ParamKind::Continuous, min: 0.0, max: 0.75, default: 0.25 },
    ParamDesc { name: "f",  kind: ParamKind::Continuous, min: 0.0, max: 0.3,  default: 0.10 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.05, default: 0.01 },
];

pub struct Aizawa {
    state:  Vec3,
    params: [f32; 7],
}

impl Aizawa {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.0, 0.0),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default,
                DESCS[4].default, DESCS[5].default, DESCS[6].default,
            ],
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
        let (a, b, c, d, e, f, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3],
            self.params[4], self.params[5], self.params[6],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * ((z - b) * x - d * y);
        self.state.y = y + dt * (d * x + (z - b) * y);
        self.state.z = z + dt * (c + a * z - (z * z * z) / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x * x * x);
    }
}

impl Default for Aizawa {
    fn default() -> Self { Self::new() }
}

impl Attractor for Aizawa {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[6].max(1e-9);
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
