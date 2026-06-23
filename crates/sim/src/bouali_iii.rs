use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Bouali Attraktor Typ III (Bouali III)
// Source: Bouali-III-1.txt (Juergen Meier, 3d-meier.de) — canonical preset
// among five parameter presets (Bouali-III-1..5.txt all share this exact
// equation set; only d, dt, N, and initial conditions differ between them).
//
// Params: [a, b, c, d, dt]
// Equations (Euler form):
//   x' = x + dt*( a*x*(1-y) - b*z )
//   y' = y + dt*( -c*y*(1-x*x) )
//   z' = z + dt*( d*x )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",  kind: ParamKind::Continuous, min: 0.0,   max: 6.0,    default: 3.0 },
    ParamDesc { name: "b",  kind: ParamKind::Continuous, min: 0.0,   max: 5.0,    default: 2.2 },
    ParamDesc { name: "c",  kind: ParamKind::Continuous, min: 0.0,   max: 3.0,    default: 1.0 },
    ParamDesc { name: "d",  kind: ParamKind::Continuous, min: 0.0,   max: 2.0,    default: 0.001 },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0,   max: 0.01,   default: 0.001 },
];

pub struct BoualiIii {
    state:  Vec3,
    params: [f32; 5],
}

impl BoualiIii {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(1.0, 1.0, 0.0),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default,
                DESCS[3].default, DESCS[4].default,
            ],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        // Original COFFEE script uses dt=0.001 and discards ~1 point before
        // recording; with our coarser per-frame stepping a few thousand Euler
        // steps reach the attractor manifold reliably.
        for _ in 0..4000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(1.0, 1.0, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, c, d, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (a * x * (1.0 - y) - b * z);
        self.state.y = y + dt * (-c * y * (1.0 - x * x));
        self.state.z = z + dt * (d * x);
    }
}

impl Default for BoualiIii {
    fn default() -> Self { Self::new() }
}

impl Attractor for BoualiIii {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[4].max(1e-9);
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(1.0, 1.0, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
