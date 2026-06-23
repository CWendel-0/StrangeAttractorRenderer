use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [alpha, beta, dT]
// Multi-segment Chua nonlinearity f(x) with 6 fixed breakpoints (not exposed
// as user parameters in the original COFFEE script either):
//   m = [-1/7, 2/7, -4/7, 2/7, -4/7, 2/7]
//   c = [0.0, 1.0, 2.15, 3.6, 8.2, 13.0]
//   f(x) = m[5]*x + 0.5 * sum_{k=1..5} (m[k-1]-m[k]) * (|x+c[k]| - |x-c[k]|)
//
// Equations (Euler form):
//   x' = x + dt*(alpha * (y - f(x)))
//   y' = y + dt*(x - y + z)
//   z' = z + dt*(-beta * y)
const M: [f32; 6] = [-1.0 / 7.0, 2.0 / 7.0, -4.0 / 7.0, 2.0 / 7.0, -4.0 / 7.0, 2.0 / 7.0];
const C: [f32; 6] = [0.0, 1.0, 2.15, 3.6, 8.2, 13.0];

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "alpha", kind: ParamKind::Continuous, min: 0.0, max: 27.0,  default: 9.0    },
    ParamDesc { name: "beta",  kind: ParamKind::Continuous, min: 0.0, max: 42.86, default: 14.286 },
    ParamDesc { name: "dT",    kind: ParamKind::Continuous, min: 0.0, max: 0.01,  default: 0.001  },
];

fn f_nonlinear(x: f32) -> f32 {
    let mut sum = 0.0;
    for k in 1..6 {
        sum += (M[k - 1] - M[k]) * ((x + C[k]).abs() - (x - C[k]).abs());
    }
    M[5] * x + 0.5 * sum
}

pub struct MultiChuaII {
    state:  Vec3,
    params: [f32; 3],
}

impl MultiChuaII {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, -0.2, 0.3),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, -0.2, 0.3);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (alpha, beta, dt) = (self.params[0], self.params[1], self.params[2]);
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let dx = alpha * (y - f_nonlinear(x));
        let dy = x - y + z;
        let dz = -beta * y;
        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for MultiChuaII {
    fn default() -> Self { Self::new() }
}

impl Attractor for MultiChuaII {
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
        self.state = Vec3::new(0.1, -0.2, 0.3);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
