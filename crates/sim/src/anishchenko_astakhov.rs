use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [m, g, dt]
// Equations (Euler form), with step function f(x) = 1 if x>0 else 0:
//   x' = x + dt*( m*x + y - x*z )
//   y' = y + dt*( -x )
//   z' = z + dt*( -g*z + g*f(x)*x^2 )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "m",  kind: ParamKind::Continuous, min: 0.0,  max: 3.0,  default: 1.2  },
    ParamDesc { name: "g",  kind: ParamKind::Continuous, min: 0.0,  max: 2.0,  default: 0.5  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,  max: 0.05, default: 0.01 },
];

pub struct AnishchenkoAstakhov {
    state:  Vec3,
    params: [f32; 3],
}

impl AnishchenkoAstakhov {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, 1.0, 1.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, 1.0, 1.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (m, g, dt) = (self.params[0], self.params[1], self.params[2]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let f = if x > 0.0 { 1.0 } else { 0.0 };
        let dx = m * x + y - x * z;
        let dy = -x;
        let dz = -g * z + g * f * x * x;
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for AnishchenkoAstakhov {
    fn default() -> Self { Self::new() }
}

impl Attractor for AnishchenkoAstakhov {
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
        self.state = Vec3::new(1.0, 1.0, 1.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
