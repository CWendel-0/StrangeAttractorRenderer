use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, s, b]
// Equations (discrete map, xa = previous x):
//   x' = a*(1 - s*cos(2*PI*y))*xa*(1-xa)
//   y' = (y + b) mod 1
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0, max: 4.0, default: 3.277 },
    ParamDesc { name: "s", kind: ParamKind::Continuous, min: 0.0, max: 1.0, default: 0.1   },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: 0.0, max: 1.0, default: 0.618 },
];

pub struct HeagyHammel {
    state:  Vec3,
    params: [f32; 3],
}

impl HeagyHammel {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
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
        let (a, s, b) = (self.params[0], self.params[1], self.params[2]);
        let xa = self.state.x;
        let y = self.state.y;
        let nx = a * (1.0 - s * (2.0 * std::f32::consts::PI * y).cos()) * xa * (1.0 - xa);
        let ny = (y + b).rem_euclid(1.0);
        self.state.x = nx;
        self.state.y = ny;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for HeagyHammel {
    fn default() -> Self { Self::new() }
}

impl Attractor for HeagyHammel {
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
