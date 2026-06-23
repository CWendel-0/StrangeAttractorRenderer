use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d, e, f, g, dT]
// Equations (Euler form):
//   x' = x + dt*(a*y + b*x + c*y*z)
//   y' = y + dt*(d*y - z + e*x*z)
//   z' = z + dt*(f*z + g*x*y)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: -7.2,  max: 7.2,  default: 2.4   },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: -11.34, max: 0.0, default: -3.78 },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: 0.0,   max: 42.0, default: 14.0  },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: -33.0, max: 0.0,  default: -11.0 },
    ParamDesc { name: "e",  kind: ParamKind::Continuous, min: 0.0,   max: 12.0, default: 4.0   },
    ParamDesc { name: "f",  kind: ParamKind::Continuous, min: 0.0,   max: 16.74,default: 5.58  },
    ParamDesc { name: "g",  kind: ParamKind::Continuous, min: -3.0,  max: 0.0,  default: -1.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,   max: 0.01, default: 0.001 },
];

pub struct LiuChen {
    state:  Vec3,
    params: [f32; 8],
}

impl LiuChen {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, 3.0, 5.0),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default,
                DESCS[4].default, DESCS[5].default, DESCS[6].default, DESCS[7].default,
            ],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..3000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, 3.0, 5.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, c, d, e, f, g, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3],
            self.params[4], self.params[5], self.params[6], self.params[7],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (a * y + b * x + c * y * z);
        self.state.y = y + dt * (d * y - z + e * x * z);
        self.state.z = z + dt * (f * z + g * x * y);
    }
}

impl Default for LiuChen {
    fn default() -> Self { Self::new() }
}

impl Attractor for LiuChen {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[7].max(1e-9);
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(1.0, 3.0, 5.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
