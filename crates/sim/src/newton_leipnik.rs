use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, dt]
// Equations (Euler form):
//   x' = x + dt*(-a*x + y + 10*y*z)
//   y' = y + dt*(-x - 0.4*y + 5*x*z)
//   z' = z + dt*(b*z - 5*x*y)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0, max: 1.2,   default: 0.4   },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0, max: 0.525, default: 0.175 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.1,   default: 0.02  },
];

pub struct NewtonLeipnik {
    state:  Vec3,
    params: [f32; 3],
}

impl NewtonLeipnik {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.349, 0.0, -0.160),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..1000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.349, 0.0, -0.160);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, dt) = (self.params[0], self.params[1], self.params[2]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (-a * x + y + 10.0 * y * z);
        self.state.y = y + dt * (-x - 0.4 * y + 5.0 * x * z);
        self.state.z = z + dt * (b * z - 5.0 * x * y);
    }
}

impl Default for NewtonLeipnik {
    fn default() -> Self { Self::new() }
}

impl Attractor for NewtonLeipnik {
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
        self.state = Vec3::new(0.349, 0.0, -0.160);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
