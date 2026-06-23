use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d, dt]
// Equations (Euler form):
//   x' = x + dt*y
//   y' = y + dt*z
//   z' = z + dt*(a*x + b*y + c*z + d*x³)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: -3.0, max: 3.0,  default: 0.8   },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: -3.0, max: 3.0,  default: -1.1  },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: -3.0, max: 3.0,  default: -0.45 },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: -3.0, max: 3.0,  default: -1.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,  max: 0.05, default: 0.01  },
];

pub struct Coullet {
    state:  Vec3,
    params: [f32; 5],
}

impl Coullet {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.10, 0.41, 0.31),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default, DESCS[4].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        let dt = self.params[4].max(1e-9);
        let steps = ((20.0 / dt) as usize).clamp(500, 20_000);
        for _ in 0..steps {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.10, 0.41, 0.31);
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
        self.state.x = x + dt * y;
        self.state.y = y + dt * z;
        self.state.z = z + dt * (a * x + b * y + c * z + d * x * x * x);
    }
}

impl Default for Coullet {
    fn default() -> Self { Self::new() }
}

impl Attractor for Coullet {
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
        self.state = Vec3::new(0.10, 0.41, 0.31);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
