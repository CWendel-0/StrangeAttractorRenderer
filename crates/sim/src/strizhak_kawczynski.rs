use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, b1, b2, q, r, ax1, ax2, ax3, dt]
// Chemical-oscillator (Chua-like cubic nonlinearity) attractor.
// Equations (Euler form):
//   x' = x + dt*( r*(y - (x-ax1)*(x-ax2)*(x-ax3) - a) )
//   y' = y + dt*( b - b1*z - b2*x - y )
//   z' = z + dt*( q*(x - z) )
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a",   kind: ParamKind::Continuous, min: 0.0, max: 450.0,  default: 150.0 },
    ParamDesc { name: "b",   kind: ParamKind::Continuous, min: 0.0, max: 1310.0, default: 436.6 },
    ParamDesc { name: "b1",  kind: ParamKind::Continuous, min: 0.0, max: 11.1,   default: 3.714 },
    ParamDesc { name: "b2",  kind: ParamKind::Continuous, min: 0.0, max: 65.1,   default: 21.7  },
    ParamDesc { name: "q",   kind: ParamKind::Continuous, min: 0.0, max: 0.21,   default: 0.07  },
    ParamDesc { name: "r",   kind: ParamKind::Continuous, min: 0.0, max: 0.3,    default: 0.101115 },
    ParamDesc { name: "ax1", kind: ParamKind::Continuous, min: 0.0, max: 30.0,   default: 10.0 },
    ParamDesc { name: "ax2", kind: ParamKind::Continuous, min: 0.0, max: 33.0,   default: 11.0 },
    ParamDesc { name: "ax3", kind: ParamKind::Continuous, min: 0.0, max: 60.0,   default: 20.0 },
    ParamDesc { name: "dT",  kind: ParamKind::Continuous, min: 0.0, max: 0.2,    default: 0.08 },
];

pub struct StrizhakKawczynski {
    state:  Vec3,
    params: [f32; 10],
}

impl StrizhakKawczynski {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.0, 0.0),
            params: [
                DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default, DESCS[4].default,
                DESCS[5].default, DESCS[6].default, DESCS[7].default, DESCS[8].default, DESCS[9].default,
            ],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.0, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, b1, b2, q, r, ax1, ax2, ax3, dt) = (
            self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
            self.params[5], self.params[6], self.params[7], self.params[8], self.params[9],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + dt * (r * (y - (x - ax1) * (x - ax2) * (x - ax3) - a));
        self.state.y = y + dt * (b - b1 * z - b2 * x - y);
        self.state.z = z + dt * (q * (x - z));
    }
}

impl Default for StrizhakKawczynski {
    fn default() -> Self { Self::new() }
}

impl Attractor for StrizhakKawczynski {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[9].max(1e-9);
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
