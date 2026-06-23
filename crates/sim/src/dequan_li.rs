use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, c, d, e, k, f, dt]
// Equations (Euler form):
//   x' = x + dt*( a*(y-x) + d*x*z )
//   y' = y + dt*( k*x + f*y - x*z )
//   z' = z + dt*( c*z + x*y - e*x² )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 120.0,   default: 40.0  },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: 0.0, max: 5.5,     default: 1.833 },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0, max: 0.48,    default: 0.16  },
    ParamDesc { name: "e",  kind: ParamKind::Continuous, min: 0.0, max: 1.95,    default: 0.65  },
    ParamDesc { name: "k",  kind: ParamKind::Continuous, min: 0.0, max: 165.0,   default: 55.0  },
    ParamDesc { name: "f",  kind: ParamKind::Continuous, min: 0.0, max: 60.0,    default: 20.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.0005,  default: 0.0001 },
];

pub struct DequanLi {
    state:  Vec3,
    params: [f32; 7],
}

impl DequanLi {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.349, 0.0, -0.160),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default,
                DESCS[4].default, DESCS[5].default, DESCS[6].default,
            ],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..50_000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.349, 0.0, -0.160);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, c, d, e, k, f, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3],
            self.params[4], self.params[5], self.params[6],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (a * (y - x) + d * x * z);
        self.state.y = y + dt * (k * x + f * y - x * z);
        self.state.z = z + dt * (c * z + x * y - e * x * x);
    }
}

impl Default for DequanLi {
    fn default() -> Self { Self::new() }
}

impl Attractor for DequanLi {
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
        self.state = Vec3::new(0.349, 0.0, -0.160);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
