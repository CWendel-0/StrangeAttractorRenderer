use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, c]
// Equations (discrete map, coupled quadratic maps, xa = previous x):
//   x' = 1 - a*xa² + c*(y² - xa²)
//   y' = 1 - a*y²  + c*(xa² - y²)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0,  max: 3.0, default: 1.84  },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: -2.0, max: 2.0, default: -0.35 },
];

pub struct Hca {
    state:  Vec3,
    params: [f32; 2],
}

impl Hca {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, 0.5, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, 0.5, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, c) = (self.params[0], self.params[1]);
        let xa = self.state.x;
        let y = self.state.y;
        let nx = 1.0 - a * xa * xa + c * (y * y - xa * xa);
        let ny = 1.0 - a * y * y + c * (xa * xa - y * y);
        self.state.x = nx;
        self.state.y = ny;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for Hca {
    fn default() -> Self { Self::new() }
}

impl Attractor for Hca {
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
        self.state = Vec3::new(1.0, 0.5, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
