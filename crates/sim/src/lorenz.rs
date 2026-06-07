use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [a, b, c, d]  (a=sigma, b=rho, c=beta, d=dt)
// Integration follows the explicit Euler form:
//   x' = x + a*d*(y - x)
//   y' = y + d*(b*x - y - z*x)
//   z' = z + d*(x*y - c*z)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "A",  kind: ParamKind::Continuous, min: 0.0, max: 20.0, default: 16.227 },
    ParamDesc { name: "B",  kind: ParamKind::Continuous, min: 0.0, max: 20.0, default: 15.223 },
    ParamDesc { name: "C",  kind: ParamKind::Continuous, min: 0.0, max: 20.0, default: 8.018  },
    ParamDesc { name: "dT", kind: ParamKind::Continuous, min: 0.0, max: 0.5,  default: 0.049  },
];

pub struct Lorenz {
    state:  Vec3,
    params: [f32; 4],
}

impl Lorenz {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.0, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..1000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.0, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let (a, b, c, d) = (
            self.params[0],
            self.params[1],
            self.params[2],
            self.params[3],
        );
        let x = self.state.x;
        let y = self.state.y;
        let z = self.state.z;
        self.state.x = x + a * d * (y - x);
        self.state.y = y + d * (b * x - y - z * x);
        self.state.z = z + d * (x * y - c * z);
    }
}

impl Default for Lorenz {
    fn default() -> Self {
        Self::new()
    }
}

impl Attractor for Lorenz {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            return None;
        }
        let speed = (self.state - prev).length() / self.params[3];
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        assert_eq!(
            params.len(), self.params.len(),
            "Lorenz::reset called with {} params, expected {}",
            params.len(), self.params.len()
        );
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(0.1, 0.0, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] {
        DESCS
    }

    fn params(&self) -> &[f32] {
        &self.params
    }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
