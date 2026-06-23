use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b]
// Equations (discrete map, z constant for 2-D attractor):
//   x' = 1 - a*x^2 + b*(y-x)
//   y' = 1 - a*y^2 + b*(x-y)
//   z' = 0
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0, max: 2.0, default: 0.9   },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: 0.0, max: 1.0, default: 0.285 },
];

pub struct StrelkovaAnishchenko {
    state:  Vec3,
    params: [f32; 2],
}

impl StrelkovaAnishchenko {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 1.0, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 1.0, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b) = (self.params[0], self.params[1]);
        let (x, y) = (self.state.x, self.state.y);
        self.state.x = 1.0 - a * x * x + b * (y - x);
        self.state.y = 1.0 - a * y * y + b * (x - y);
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for StrelkovaAnishchenko {
    fn default() -> Self { Self::new() }
}

impl Attractor for StrelkovaAnishchenko {
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
        self.state = Vec3::new(0.1, 1.0, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
