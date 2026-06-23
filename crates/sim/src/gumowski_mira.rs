use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b]
// Equations (discrete map, z constant for 2-D attractor):
//   f(t) = b*t + 2*(1-b)*t² / (1+t²)
//   x' = y + a*(1 - 0.05*y²)*y + f(x)
//   y' = -x + f(x')
//   z' = 0
//
// Source: Gumowski-Mira.cof — a=0.008, b=-0.90, start (0.10, 0.10).
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -0.5, max: 0.5, default:  0.008 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: -1.5, max: 1.5, default: -0.90  },
];

fn f(t: f32, b: f32) -> f32 {
    b * t + (2.0 * (1.0 - b) * t * t) / (1.0 + t * t)
}

pub struct GumowskiMira {
    state:  Vec3,
    params: [f32; 2],
}

impl GumowskiMira {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.10, 0.10, 0.0),
            params: [DESCS[0].default, DESCS[1].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.10, 0.10, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b) = (self.params[0], self.params[1]);
        let (x, y) = (self.state.x, self.state.y);
        let nx = y + a * (1.0 - 0.05 * y * y) * y + f(x, b);
        let ny = -x + f(nx, b);
        self.state.x = nx;
        self.state.y = ny;
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for GumowskiMira {
    fn default() -> Self { Self::new() }
}

impl Attractor for GumowskiMira {
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
        self.state = Vec3::new(0.10, 0.10, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
