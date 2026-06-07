use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [degree, alpha, beta, lambda, gamma, omega]
//
// Equations (Field-Golubitsky Icon, original form):
//   r   = (x + iy)^d
//   ‖v‖ = sqrt(x² + y²)        (position norm; z is purely visual)
//   p   = λ + α‖v‖ + β(x·Re r − y·Im r)
//   x'  = p·x + γ·Re r − ω·y
//   y'  = p·y − γ·Im r + ω·x
//   z'  = ‖v‖

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "degree", kind: ParamKind::Integer,    min: 2.0, max: 10.0, default: 2.0   },
    ParamDesc { name: "alpha",  kind: ParamKind::Continuous, min: -4.0, max: 4.0, default: -1.715},
    ParamDesc { name: "beta",   kind: ParamKind::Continuous, min: -4.0, max: 4.0, default: -0.689},
    ParamDesc { name: "lambda", kind: ParamKind::Continuous, min: -4.0, max: 4.0, default:  2.343},
    ParamDesc { name: "gamma",  kind: ParamKind::Continuous, min: -4.0, max: 4.0, default:  0.351},
    ParamDesc { name: "omega",  kind: ParamKind::Continuous, min: -4.0, max: 4.0, default: -0.428},
];

pub struct IconB {
    state:  Vec3,
    params: [f32; 6],
}

impl IconB {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::ZERO,
            params: std::array::from_fn(|i| DESCS[i].default),
        };
        s.warm_up();
        s
    }

    fn eq_radius(&self) -> f32 {
        let (alpha, lambda) = (self.params[1], self.params[3]);
        // p equilibrium on real axis (ignoring beta): λ + α·R = 1  →  R = (1-λ)/α
        let rhs = if alpha.abs() > 1e-6 { (1.0 - lambda) / alpha } else { 0.0 };
        if rhs > 0.0 { rhs.clamp(0.01, 2.0) } else { 0.5 }
    }

    fn warm_up(&mut self) {
        let r = self.eq_radius();
        self.state = Vec3::new(r, 0.01, 0.0);
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(r, 0.01, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let degree = self.params[0].max(1.0) as u32;
        let (alpha, beta, lambda, gamma, omega) =
            (self.params[1], self.params[2], self.params[3], self.params[4], self.params[5]);
        let (x, y) = (self.state.x, self.state.y);

        let (mut re_r, mut im_r) = (1.0f32, 0.0f32);
        for _ in 0..degree {
            let nr = re_r * x - im_r * y;
            let ni = re_r * y + im_r * x;
            re_r = nr; im_r = ni;
        }

        let norm_v = (x * x + y * y).sqrt();
        let p      = lambda + alpha * norm_v + beta * (x * re_r - y * im_r);

        self.state.x = p * x + gamma * re_r - omega * y;
        self.state.y = p * y - gamma * im_r + omega * x;
        self.state.z = norm_v;
    }
}

impl Default for IconB {
    fn default() -> Self { Self::new() }
}

impl Attractor for IconB {
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
