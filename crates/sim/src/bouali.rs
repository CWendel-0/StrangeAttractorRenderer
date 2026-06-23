use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, s, dt]
// Equations (Euler form):
//   x' = x + dt*( x*(4-y) + a*z )
//   y' = y + dt*( -y*(1-x²) )
//   z' = z + dt*( -x*(1.5-s*z) - 0.05*z )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 0.9,  default: 0.3 },
    ParamDesc { name: "s",  kind: ParamKind::Continuous, min: 0.0, max: 3.0,  default: 1.0 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.03, default: 0.006 },
];

pub struct Bouali {
    state:  Vec3,
    params: [f32; 3],
}

impl Bouali {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, 0.1, 0.1),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..3300 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, 0.1, 0.1);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, s, dt) = (self.params[0], self.params[1], self.params[2]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (x * (4.0 - y) + a * z);
        self.state.y = y + dt * (-y * (1.0 - x * x));
        self.state.z = z + dt * (-x * (1.5 - s * z) - 0.05 * z);
    }
}

impl Default for Bouali {
    fn default() -> Self { Self::new() }
}

impl Attractor for Bouali {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[2].max(1e-9);
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(1.0, 0.1, 0.1);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
