use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [p0, p1, p2, p3, p4, p5]
// Equations (discrete map):
//   x' = p0 + y - z*(p1 + y)
//   y' = p2 + z - x*(p3 + z)
//   z' = p4 + x - y*(p5 + x)
//
// Poly A is the special case where p1 = p3 = p5 = 0.

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "P0", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.328 },
    ParamDesc { name: "P1", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.53  },
    ParamDesc { name: "P2", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.404 },
    ParamDesc { name: "P3", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.188 },
    ParamDesc { name: "P4", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.642 },
    ParamDesc { name: "P5", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.89  },
];

pub struct PolyB {
    state:  Vec3,
    params: [f32; 6],
}

impl PolyB {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.1),
            params: std::array::from_fn(|i| DESCS[i].default),
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.1, 0.1);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let p = &self.params;
        let (x, y, z) = (self.state.x, self.state.y, self.state.z);
        self.state.x = p[0] + y - z * (p[1] + y);
        self.state.y = p[2] + z - x * (p[3] + z);
        self.state.z = p[4] + x - y * (p[5] + x);
    }
}

impl Default for PolyB {
    fn default() -> Self { Self::new() }
}

impl Attractor for PolyB {
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
        self.state = Vec3::new(0.1, 0.1, 0.1);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
