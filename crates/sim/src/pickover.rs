use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d]
// Equations (discrete map):
//   x' = sin(a·y) - z·cos(b·x)
//   y' = z·sin(c·x) - cos(d·y)
//   z' = sin(x)

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: -3.0, max: 3.0, default: -2.24 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: -3.0, max: 3.0, default:  0.43 },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: -3.0, max: 3.0, default: -0.65 },
    ParamDesc { name: "d", kind: ParamKind::Continuous, min: -3.0, max: 3.0, default: -2.43 },
];

pub struct Pickover {
    state:  Vec3,
    params: [f32; 4],
}

impl Pickover {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.0, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.0, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (a, b, c, d) = (self.params[0], self.params[1], self.params[2], self.params[3]);
        let (x, y, z) = (self.state.x, self.state.y, self.state.z);
        self.state.x = (a * y).sin() - z * (b * x).cos();
        self.state.y = z * (c * x).sin() - (d * y).cos();
        self.state.z = x.sin();
    }
}

impl Default for Pickover {
    fn default() -> Self { Self::new() }
}

impl Attractor for Pickover {
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
        self.state = Vec3::new(0.1, 0.0, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
