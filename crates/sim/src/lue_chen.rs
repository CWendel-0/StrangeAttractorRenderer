use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, dt]
// Equations (Euler form):
//   x' = x + dt*( -a*b*x/(a+b) - y*z + c )
//   y' = y + dt*( a*y + x*z )
//   z' = z + dt*( b*z + x*y )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: -20.0, max: 0.0,  default: -10.0 },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: -10.0, max: 0.0,  default: -4.0   },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: 0.0,   max: 40.0, default: 18.1   },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,   max: 0.05, default: 0.01   },
];

pub struct LueChen {
    state:  Vec3,
    params: [f32; 4],
}

impl LueChen {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.0, 0.0, 2.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.0, 0.0, 2.0);
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
        let denom = if (a + b).abs() < 1e-9 { 1e-9 } else { a + b };
        let dx = -a * b * x / denom - y * z + c;
        let dy = a * y + x * z;
        let dz = b * z + x * y;
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for LueChen {
    fn default() -> Self { Self::new() }
}

impl Attractor for LueChen {
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
        self.state = Vec3::new(0.0, 0.0, 2.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
