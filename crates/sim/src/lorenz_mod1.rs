use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d, dt]
// Equations (Euler form):
//   x' = x + dt*( -a*x + y^2 - z^2 + a*c )
//   y' = y + dt*( x*(y - b*z) + d )
//   z' = z + dt*( z + x*(b*y + z) )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0,  max: 2.0,  default: 0.1  },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0,  max: 10.0, default: 4.0  },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: 0.0,  max: 30.0, default: 14.0 },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0,  max: 1.0,  default: 0.08 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,  max: 0.05, default: 0.008},
];

pub struct LorenzMod1 {
    state:  Vec3,
    params: [f32; 5],
}

impl LorenzMod1 {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.10, 0.10, 0.10),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default, DESCS[4].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2500 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.10, 0.10, 0.10);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, c, d, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let dx = -a * x + y * y - z * z + a * c;
        let dy = x * (y - b * z) + d;
        let dz = z + x * (b * y + z);
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for LorenzMod1 {
    fn default() -> Self { Self::new() }
}

impl Attractor for LorenzMod1 {
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
