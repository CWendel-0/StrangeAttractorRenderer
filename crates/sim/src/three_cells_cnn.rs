use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [p1, p2, r, s, dt]
// 3-cell CNN (cellular neural network) attractor. h1/h2/h3 are saturation
// nonlinearities of x/y/z (a piecewise-linear "soft sign"), recomputed each
// step rather than separate state.
//   h(v) = 0.5*(|v+1| - |v-1|)
// Equations (Euler form):
//   x' = x + dt*(-x + p1*h(x) - s*h(y) - s*h(z))
//   y' = y + dt*(-y - s*h(x) + p2*h(y) - r*h(z))
//   z' = z + dt*(-z - s*h(x) + r*h(y) + h(z))
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "p1", kind: ParamKind::Continuous, min: 0.0, max: 3.72,  default: 1.24 },
    ParamDesc { name: "p2", kind: ParamKind::Continuous, min: 0.0, max: 3.3,   default: 1.1  },
    ParamDesc { name: "r",  kind: ParamKind::Continuous, min: 0.0, max: 13.2,  default: 4.4  },
    ParamDesc { name: "s",  kind: ParamKind::Continuous, min: 0.0, max: 9.63,  default: 3.21 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.035, default: 0.007 },
];

fn h(v: f32) -> f32 { 0.5 * ((v + 1.0).abs() - (v - 1.0).abs()) }

pub struct ThreeCellsCnn {
    state:  Vec3,
    params: [f32; 5],
}

impl ThreeCellsCnn {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.1),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default, DESCS[4].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..2900 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.1, 0.1);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (p1, p2, r, s, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        let (hx, hy, hz) = (h(x), h(y), h(z));
        self.state.x = x + dt * (-x + p1 * hx - s * hy - s * hz);
        self.state.y = y + dt * (-y - s * hx + p2 * hy - r * hz);
        self.state.z = z + dt * (-z - s * hx + r * hy + hz);
    }
}

impl Default for ThreeCellsCnn {
    fn default() -> Self { Self::new() }
}

impl Attractor for ThreeCellsCnn {
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
        self.state = Vec3::new(0.1, 0.1, 0.1);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
