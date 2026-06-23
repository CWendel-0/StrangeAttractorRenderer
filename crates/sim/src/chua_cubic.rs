use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [alpha, beta, c, dt]
// Equations (Euler form), continuous-time ODE flow:
//   x' = x + dt*( alpha * (y - x³ - c*x) )
//   y' = y + dt*( x - y + z )
//   z' = z + dt*( -beta * y )
//
// Source: Chua_Cubic.cof — alpha=10.0, beta=16.0, c=-0.143, Delta=0.005,
// start (0.1, 0.1, 0.1).
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "alpha", kind: ParamKind::Continuous, min: 0.0,   max: 20.0,  default: 10.0   },
    ParamDesc { name: "beta",  kind: ParamKind::Continuous, min: 0.0,   max: 30.0,  default: 16.0   },
    ParamDesc { name: "c",     kind: ParamKind::Continuous, min: -2.0,  max: 2.0,   default: -0.143 },
    ParamDesc { name: "dT",    kind: ParamKind::Continuous, min: 0.0,   max: 0.02,  default: 0.005  },
];

pub struct ChuaCubic {
    state:  Vec3,
    params: [f32; 4],
}

impl ChuaCubic {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.1),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..4000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.1, 0.1);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (alpha, beta, c, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (alpha * (y - x * x * x - c * x));
        self.state.y = y + dt * (x - y + z);
        self.state.z = z + dt * (-beta * y);
    }
}

impl Default for ChuaCubic {
    fn default() -> Self { Self::new() }
}

impl Attractor for ChuaCubic {
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
        self.state = Vec3::new(0.1, 0.1, 0.1);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
