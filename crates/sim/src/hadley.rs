use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, f, g, dt]
// Equations (Euler form):
//   x' = x + dt*( -y² - z² - a*x + a*f )
//   y' = y + dt*( x*y - b*x*z - y + g )
//   z' = z + dt*( b*x*y + x*z - z )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 0.6,   default: 0.20 },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0, max: 12.0,  default: 4.0  },
    ParamDesc { name: "f",  kind: ParamKind::Continuous, min: 0.0, max: 24.0,  default: 8.0  },
    ParamDesc { name: "g",  kind: ParamKind::Continuous, min: 0.0, max: 3.0,   default: 1.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.025, default: 0.005 },
];

pub struct Hadley {
    state:  Vec3,
    params: [f32; 5],
}

impl Hadley {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.0, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default, DESCS[4].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..4000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.0, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, f, g, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (-y * y - z * z - a * x + a * f);
        self.state.y = y + dt * (x * y - b * x * z - y + g);
        self.state.z = z + dt * (b * x * y + x * z - z);
    }
}

impl Default for Hadley {
    fn default() -> Self { Self::new() }
}

impl Attractor for Hadley {
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
        self.state = Vec3::new(0.1, 0.0, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
