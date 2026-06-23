use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d, e, f, dt]
// Equations (Euler form):
//   x' = x + dt*(a*x + c*y*z)
//   y' = y + dt*(b*x + d*y - x*z)
//   z' = z + dt*(e*z + f*x*y)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: -2.0, max: 2.0,  default: 0.2  },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: -2.0, max: 2.0,  default: -0.01 },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: -2.0, max: 2.0,  default: 1.0  },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: -2.0, max: 2.0,  default: -0.4 },
    ParamDesc { name: "e",  kind: ParamKind::Continuous, min: -2.0, max: 2.0,  default: -1.0 },
    ParamDesc { name: "f",  kind: ParamKind::Continuous, min: -2.0, max: 2.0,  default: -1.0 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,  max: 0.05, default: 0.01 },
];

pub struct WangSun {
    state:  Vec3,
    params: [f32; 7],
}

impl WangSun {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.30, 0.10, 1.00),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default,
                DESCS[4].default, DESCS[5].default, DESCS[6].default,
            ],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        let dt = self.params[6].max(1e-9);
        let steps = ((20.0 / dt) as usize).clamp(500, 20_000);
        for _ in 0..steps {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.30, 0.10, 1.00);
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
        self.state.x = x + dt * (a * x + c * y * z);
        self.state.y = y + dt * (b * x + d * y - x * z);
        self.state.z = z + dt * (e * z + f * x * y);
    }
}

impl Default for WangSun {
    fn default() -> Self { Self::new() }
}

impl Attractor for WangSun {
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
        self.state = Vec3::new(0.30, 0.10, 1.00);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
