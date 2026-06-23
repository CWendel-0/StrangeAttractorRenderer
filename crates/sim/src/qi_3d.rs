use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d, e, dt]
// Equations (Euler form):
//   x' = x + dt*( a*(y-x) + e*y*z )
//   y' = y + dt*( c*x + d*y - x*z )
//   z' = z + dt*( x*y - b*z )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0,   max: 30.0, default: 16.0  },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0,   max: 60.0, default: 43.0  },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: -30.0, max: 0.0,  default: -18.0 },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0,   max: 30.0, default: 16.0  },
    ParamDesc { name: "e",  kind: ParamKind::Continuous, min: -3.0,  max: 0.0,  default: -1.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,   max: 0.01, default: 0.001 },
];

pub struct Qi3D {
    state:  Vec3,
    params: [f32; 6],
}

impl Qi3D {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(3.0, -4.0, 3.0),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default,
                DESCS[3].default, DESCS[4].default, DESCS[5].default,
            ],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..20000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(3.0, -4.0, 3.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, c, d, e, dt) = (
            self.params[0], self.params[1], self.params[2],
            self.params[3], self.params[4], self.params[5],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let dx = a * (y - x) + e * y * z;
        let dy = c * x + d * y - x * z;
        let dz = x * y - b * z;
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for Qi3D {
    fn default() -> Self { Self::new() }
}

impl Attractor for Qi3D {
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
        self.state = Vec3::new(3.0, -4.0, 3.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
