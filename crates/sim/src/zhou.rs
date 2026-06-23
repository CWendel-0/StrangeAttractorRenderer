use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, dt]
// Equations (Euler form, Lorenz-like flow):
//   x' = x + dt*a*(y - x)
//   y' = y + dt*(b*x - x*z)
//   z' = z + dt*(x*y + c*z)
// Original constants: a=10.0, b=16.0, c=-1.0, Delta=0.002.
// Original initial values: x0=3.0, y0=1.0, z0=15.0.

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0,  max: 30.0, default: 10.0  },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0,  max: 40.0, default: 16.0  },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: -5.0, max: 5.0,  default: -1.0  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,  max: 0.01, default: 0.002 },
];

pub struct Zhou {
    state:  Vec3,
    params: [f32; 4],
}

impl Zhou {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(3.0, 1.0, 15.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        // ~20-30 time units worth of steps at the default dt.
        let dt = self.params[3].max(1e-6);
        let steps = (20.0 / dt) as u32;
        let steps = steps.clamp(500, 20000);
        for _ in 0..steps {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(3.0, 1.0, 15.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, c, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (a * (y - x));
        self.state.y = y + dt * (b * x - x * z);
        self.state.z = z + dt * (x * y + c * z);
    }
}

impl Default for Zhou {
    fn default() -> Self { Self::new() }
}

impl Attractor for Zhou {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[3].max(1e-9);
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(3.0, 1.0, 15.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
