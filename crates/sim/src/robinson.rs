use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d, v, dT]
// Equations (Euler form):
//   x' = x + dt*y
//   y' = y + dt*(x - 2*x³ - a*y + b*x²*y - v*y*z)
//   z' = z + dt*(-c*z + d*x²)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 2.13,  default: 0.71   },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0, max: 5.58,  default: 1.8587 },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: 0.0, max: 2.12,  default: 0.7061 },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0, max: 0.3,   default: 0.1    },
    ParamDesc { name: "v",  kind: ParamKind::Continuous, min: 0.0, max: 3.0,   default: 1.0    },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.05,  default: 0.01   },
];

pub struct Robinson {
    state:  Vec3,
    params: [f32; 6],
}

impl Robinson {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.10, 0.10, 0.10),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default,
                DESCS[3].default, DESCS[4].default, DESCS[5].default,
            ],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.10, 0.10, 0.10);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, c, d, v, dt) = (
            self.params[0], self.params[1], self.params[2],
            self.params[3], self.params[4], self.params[5],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let dx = y;
        let dy = x - 2.0 * x * x * x - a * y + b * x * x * y - v * y * z;
        let dz = -c * z + d * x * x;
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for Robinson {
    fn default() -> Self { Self::new() }
}

impl Attractor for Robinson {
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
        self.state = Vec3::new(0.10, 0.10, 0.10);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
