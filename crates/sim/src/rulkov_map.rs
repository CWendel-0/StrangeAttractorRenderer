use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c]
// Equations (discrete map, z constant for 2-D attractor; xa = previous x):
//   x' = a / (1 + xa²) + y
//   y' = y - b·xa - c
//   z' = 0
//
// Source: Rulkov-Map.txt (Juergen Meier).

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0,   max: 8.0,   default: 4.005 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: -0.1,  max: 0.1,   default: 0.004 },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: -0.1,  max: 0.1,   default: 0.004 },
];

pub struct RulkovMap {
    state:  Vec3,
    params: [f32; 3],
}

impl RulkovMap {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.1, 1.1, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.1, 1.1, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b, c) = (self.params[0], self.params[1], self.params[2]);
        let (xa, y) = (self.state.x, self.state.y);
        self.state.x = a / (1.0 + xa * xa) + y;
        self.state.y = y - b * xa - c;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for RulkovMap {
    fn default() -> Self { Self::new() }
}

impl Attractor for RulkovMap {
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
        self.state = Vec3::new(1.1, 1.1, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
