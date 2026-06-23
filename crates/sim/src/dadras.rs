use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [p, q, r, s, e, dt]
// Equations (Euler form):
//   x' = x + dt*( y - p*x + q*y*z )
//   y' = y + dt*( r*y - x*z + z )
//   z' = z + dt*( s*x*y - e*z )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "p",  kind: ParamKind::Continuous, min: 0.0,  max: 8.0,  default: 3.0  },
    ParamDesc { name: "q",  kind: ParamKind::Continuous, min: 0.0,  max: 8.0,  default: 2.7  },
    ParamDesc { name: "r",  kind: ParamKind::Continuous, min: 0.0,  max: 8.0,  default: 1.7  },
    ParamDesc { name: "s",  kind: ParamKind::Continuous, min: 0.0,  max: 8.0,  default: 2.0  },
    ParamDesc { name: "e",  kind: ParamKind::Continuous, min: 0.0,  max: 20.0, default: 9.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,  max: 0.05, default: 0.01 },
];

pub struct Dadras {
    state:  Vec3,
    params: [f32; 6],
}

impl Dadras {
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
        let (p, q, r, s, e, dt) = (
            self.params[0], self.params[1], self.params[2],
            self.params[3], self.params[4], self.params[5],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let dx = y - p * x + q * y * z;
        let dy = r * y - x * z + z;
        let dz = s * x * y - e * z;
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for Dadras {
    fn default() -> Self { Self::new() }
}

impl Attractor for Dadras {
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
