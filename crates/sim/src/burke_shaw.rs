use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [s, v, dt]
// Equations (Euler form):
//   x' = x + dt*(-s*(x+y))
//   y' = y + dt*(-y - s*x*z)
//   z' = z + dt*(s*x*y + v)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "s",  kind: ParamKind::Continuous, min: 0.0, max: 30.0,  default: 10.0  },
    ParamDesc { name: "v",  kind: ParamKind::Continuous, min: 0.0, max: 12.8,  default: 4.272 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.025, default: 0.005 },
];

pub struct BurkeShaw {
    state:  Vec3,
    params: [f32; 3],
}

impl BurkeShaw {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, 0.0, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..4000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, 0.0, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (s, v, dt) = (self.params[0], self.params[1], self.params[2]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (-s * (x + y));
        self.state.y = y + dt * (-y - s * x * z);
        self.state.z = z + dt * (s * x * y + v);
    }
}

impl Default for BurkeShaw {
    fn default() -> Self { Self::new() }
}

impl Attractor for BurkeShaw {
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
        self.state = Vec3::new(1.0, 0.0, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
