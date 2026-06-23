use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d, e, f, dt]
// Equations (Euler form):
//   x' = x + dt*(a*x + b*y + c*y*z)
//   y' = y + dt*(d*y - x*z)
//   z' = z + dt*(e*z + f*x*y)
//
// Source: Four-Wing2.cof — a=-14.0, b=5.0, c=1.0, d=16.0, e=-43.0, f=1.0,
// Delta=0.0005, start (4.0, 1.0, 1.0).
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: -30.0, max: 0.0,   default: -14.0  },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0,   max: 15.0,  default: 5.0    },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: -5.0,  max: 5.0,   default: 1.0    },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0,   max: 30.0,  default: 16.0   },
    ParamDesc { name: "e",  kind: ParamKind::Continuous, min: -60.0, max: 0.0,   default: -43.0  },
    ParamDesc { name: "f",  kind: ParamKind::Continuous, min: -5.0,  max: 5.0,   default: 1.0    },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,   max: 0.005, default: 0.0005 },
];

pub struct FourWing2 {
    state:  Vec3,
    params: [f32; 7],
}

impl FourWing2 {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(4.0, 1.0, 1.0),
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
                self.state = Vec3::new(4.0, 1.0, 1.0);
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
        self.state.x = x + dt * (a * x + b * y + c * y * z);
        self.state.y = y + dt * (d * y - x * z);
        self.state.z = z + dt * (e * z + f * x * y);
    }
}

impl Default for FourWing2 {
    fn default() -> Self { Self::new() }
}

impl Attractor for FourWing2 {
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
        self.state = Vec3::new(4.0, 1.0, 1.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
