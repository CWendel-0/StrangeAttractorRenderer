use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, d, m, dt]
// Equations (Euler form):
//   x' = x + dt*( a*(x-y) )
//   y' = y + dt*( -4*a*y + x*z + m*x^3 )
//   z' = z + dt*( -d*a*z + x*y + b*z^2 )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0,   max: 5.0,   default: 1.8   },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: -1.0,  max: 1.0,   default: -0.07 },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0,   max: 5.0,   default: 1.5    },
    ParamDesc { name: "m",  kind: ParamKind::Continuous, min: -1.0,  max: 1.0,   default: 0.02   },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,   max: 0.05,  default: 0.01   },
];

pub struct Act {
    state:  Vec3,
    params: [f32; 5],
}

impl Act {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.10, 0.10, 0.10),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default, DESCS[4].default],
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
        let (a, b, d, m, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let dx = a * (x - y);
        let dy = -4.0 * a * y + x * z + m * x * x * x;
        let dz = -d * a * z + x * y + b * z * z;
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for Act {
    fn default() -> Self { Self::new() }
}

impl Attractor for Act {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[4].max(1e-9);
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
