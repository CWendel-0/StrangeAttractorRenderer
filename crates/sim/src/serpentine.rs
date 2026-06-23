use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a]
// Equations (discrete map, z constant for 2-D attractor):
//   k  = 2^a
//   x' = atan(cot(k*x)) = atan(1 / tan(k*x))
//   y' = sin(k*y) * cos(k*y)
//   z' = 0
// Original initial conditions: x0 = 0.10, y0 = 0.10

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0, max: 4.0, default: 1.9 },
];

pub struct Serpentine {
    state:  Vec3,
    params: [f32; 1],
}

impl Serpentine {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.10, 0.10, 0.0),
            params: [DESCS[0].default],
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
        let a = self.params[0];
        let k = 2.0_f32.powf(a);
        let (x, y) = (self.state.x, self.state.y);
        // cot(theta) = 1 / tan(theta)
        self.state.x = (1.0 / (k * x).tan()).atan();
        self.state.y = (k * y).sin() * (k * y).cos();
        // z stays 0; camera looks down z-axis for 2-D maps
    }
}

impl Default for Serpentine {
    fn default() -> Self { Self::new() }
}

impl Attractor for Serpentine {
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
