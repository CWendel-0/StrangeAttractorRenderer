use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [degree, alpha, beta, lambda, gamma, omega]
//   degree — Integer; exponent of the complex power
// Equations:
//   r  = (x + iy)^degree   (complex power)
//   p  = λ + α|r| + β(x·Re(r) - y·Im(r))
//   x' = p·x + γ·Re(r) - ω·Im(r)
//   y' = p·y - γ·Im(r) + ω·Re(r)
//   z' = sqrt(x² + y²)    (radius as 3-D depth channel)

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "degree", kind: ParamKind::Integer,    min: 2.0, max: 10.0, default: 7.0   },
    ParamDesc { name: "alpha",  kind: ParamKind::Continuous, min: -4.0, max: 4.0, default: -2.535},
    ParamDesc { name: "beta",   kind: ParamKind::Continuous, min: -4.0, max: 4.0, default: -0.669},
    ParamDesc { name: "lambda", kind: ParamKind::Continuous, min: -4.0, max: 4.0, default:  2.402},
    ParamDesc { name: "gamma",  kind: ParamKind::Continuous, min: -4.0, max: 4.0, default:  0.179},
    ParamDesc { name: "omega",  kind: ParamKind::Continuous, min: -4.0, max: 4.0, default: -0.452},
];

pub struct Icon {
    state:  Vec3,
    params: [f32; 6],
}

impl Icon {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.5, 0.3, 0.0),
            params: [DESCS[0].default, DESCS[1].default, DESCS[2].default,
                     DESCS[3].default, DESCS[4].default, DESCS[5].default],
        };
        s.warm_up();
        s
    }

    fn icon_eq_radius(&self) -> f32 {
        let (alpha, lambda) = (self.params[1], self.params[3]);
        // Standard Field-Golubitsky form uses alpha*|z|^2, so equilibrium: R^2 = (1-lambda)/alpha
        let rhs = if alpha.abs() > 1e-6 { (1.0 - lambda) / alpha } else { 0.0 };
        if rhs > 0.0 { rhs.sqrt().clamp(0.01, 2.0) } else { 0.5 }
    }

    fn warm_up(&mut self) {
        let r = self.icon_eq_radius();
        self.state = Vec3::new(r, 0.0, 0.0);
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(r, 0.0, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let degree = self.params[0].max(1.0) as u32;
        let (alpha, beta, lambda, gamma, omega) =
            (self.params[1], self.params[2], self.params[3], self.params[4], self.params[5]);
        let (x, y) = (self.state.x, self.state.y);

        // Accumulate z^k; track z^(degree-1) for the conj(z)^(degree-1) term
        let (mut re_r, mut im_r) = (1.0f32, 0.0f32);
        let (mut re_prev, mut im_prev) = (1.0f32, 0.0f32);
        for _ in 0..degree {
            re_prev = re_r; im_prev = im_r;
            let nr = re_r * x - im_r * y;
            let ni = re_r * y + im_r * x;
            re_r = nr; im_r = ni;
        }
        // re_r/im_r = z^degree, re_prev/im_prev = z^(degree-1)

        // Standard Field-Golubitsky Icon map:
        //   p = λ + α|z|² + β·Re(z^l)
        //   z' = p·z + (γ+iω)·conj(z)^(l-1)
        let r_sq = x * x + y * y;
        let p = lambda + alpha * r_sq + beta * re_r;
        // conj(z)^(l-1) = conj(z^(l-1)) = re_prev - i·im_prev
        // (γ+iω)(re_prev - i·im_prev) → real: γ·re_prev + ω·im_prev, imag: ω·re_prev - γ·im_prev
        self.state.x = p * x + gamma * re_prev + omega * im_prev;
        self.state.y = p * y + omega * re_prev - gamma * im_prev;
        self.state.z = r_sq.sqrt();
    }
}

impl Default for Icon {
    fn default() -> Self { Self::new() }
}

impl Attractor for Icon {
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
        self.state = Vec3::new(0.5, 0.3, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
