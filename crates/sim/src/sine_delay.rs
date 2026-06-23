use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b]
// 3rd-order recursion delay-embedded into 2-D (Jurgen Meier "Sine Delay Attraktor"):
//   x[n+1] = b*x[n-1] + a*sin(x[n])
// Plotted point: (x[n-1], x[n]) before the update, i.e. state (u, v) with
//   u' = v
//   v' = b*u + a*sin(v)
// Original initial values: x0 = 0.1, x1 = 0.1.

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0, max: 6.0, default: 3.077 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: -2.0, max: 2.0, default: 0.3 },
];

pub struct SineDelay {
    state:  Vec3, // x = u (x[n-1]), y = v (x[n]), z = 0
    params: [f32; 2],
}

impl SineDelay {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.1, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b) = (self.params[0], self.params[1]);
        let (u, v) = (self.state.x, self.state.y);
        let v_next = b * u + a * v.sin();
        self.state.x = v;
        self.state.y = v_next;
    }
}

impl Default for SineDelay {
    fn default() -> Self { Self::new() }
}

impl Attractor for SineDelay {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.map_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length();
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(0.1, 0.1, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
