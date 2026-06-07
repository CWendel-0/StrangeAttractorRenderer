use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [p0, p1, p2]
// Equations (discrete map):
//   x' = p0 + y - z*y
//   y' = p1 + z - x*z
//   z' = p2 + x - y*x
//
// Special case of Poly B where p1, p3, p5 = 0.

// Parameters near 0 or 2 cause the corresponding coordinate to collapse toward
// a fixed value, producing near-degenerate maps.  All known interesting PolyA
// attractors (including Chaoscope examples) have parameters ≥ 0.15.
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "P0", kind: ParamKind::Continuous, min: 0.15, max: 1.85, default: 1.092 },
    ParamDesc { name: "P1", kind: ParamKind::Continuous, min: 0.15, max: 1.85, default: 0.223 },
    ParamDesc { name: "P2", kind: ParamKind::Continuous, min: 0.15, max: 1.85, default: 1.612 },
];

pub struct PolyA {
    state:  Vec3,
    params: [f32; 3],
}

impl PolyA {
    pub fn new() -> Self {
        let mut s = Self {
            // (1,1,1) always maps to (p0,p1,p2) for this cyclic map, landing
            // inside the bounded region regardless of parameter values.
            state:  Vec3::ONE,
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        self.state = Vec3::ONE;
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() || self.state.length_squared() > 1e10 {
                self.state = Vec3::ONE;
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let (p0, p1, p2) = (self.params[0], self.params[1], self.params[2]);
        let (x, y, z) = (self.state.x, self.state.y, self.state.z);
        self.state.x = p0 + y - z * y;
        self.state.y = p1 + z - x * z;
        self.state.z = p2 + x - y * x;
    }
}

impl Default for PolyA {
    fn default() -> Self { Self::new() }
}

impl Attractor for PolyA {
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
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
