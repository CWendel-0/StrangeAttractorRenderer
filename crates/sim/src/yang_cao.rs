use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Yang-Cao (Juergen Meier, 3d-meier.de, "Yang Cao.cof")
// Coupled 2-D map in (w, s); the plotted point (x, y) is a derived
// projection of the freshly-updated (w, s) state each iteration.
//
// Params: [a, b, c, u, m]
// Equations (discrete map, z constant):
//   t  = c - 6/(1 + w^2 + s^2)
//   w' = 1 + u*(w*cos(t) - s*sin(t))
//   s' = u*(w*sin(t) + s*cos(t))
//   x  = 1 - a*s'^2 + b*w'
//   y  = m*w'*(1 - s')
//   plotted state: (x, y)
const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "a", kind: ParamKind::Continuous, min: 0.0, max: 5.0, default: 1.30 },
    ParamDesc { name: "b", kind: ParamKind::Continuous, min: 0.0, max: 10.0, default: 4.00 },
    ParamDesc { name: "c", kind: ParamKind::Continuous, min: -5.0, max: 5.0, default: 0.40 },
    ParamDesc { name: "u", kind: ParamKind::Continuous, min: 0.0, max: 2.0, default: 0.90 },
    ParamDesc { name: "m", kind: ParamKind::Continuous, min: 0.0, max: 10.0, default: 4.00 },
];

pub struct YangCao {
    state:     Vec3, // x = w, y = s  (the true map state); z unused (0)
    last_plot: Vec3, // previous derived (x, y) plot point, for speed calc
    params:    [f32; 5],
}

impl YangCao {
    pub fn new() -> Self {
        let mut s = Self {
            state:     Vec3::new(0.10, 0.10, 0.0),
            last_plot: Vec3::ZERO,
            params:    [DESCS[0].default, DESCS[1].default, DESCS[2].default, DESCS[3].default, DESCS[4].default],
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            let plot = self.map_step();
            if !self.state.is_finite() || !plot.is_finite() {
                self.state = Vec3::new(0.10, 0.10, 0.0);
                self.last_plot = Vec3::ZERO;
                break;
            }
            self.last_plot = plot;
        }
    }

    /// Advances the (w, s) state by one iteration and returns the derived
    /// plotted point (x, y) computed from the freshly-updated (w, s).
    fn map_step(&mut self) -> Vec3 {
        let (a, b, c, u, m) = (self.params[0], self.params[1], self.params[2], self.params[3], self.params[4]);
        let (w, s) = (self.state.x, self.state.y);
        let t = c - 6.0 / (1.0 + w * w + s * s);
        let w2 = 1.0 + u * (w * t.cos() - s * t.sin());
        let s2 = u * (w * t.sin() + s * t.cos());
        self.state.x = w2;
        self.state.y = s2;
        let x = 1.0 - a * s2 * s2 + b * w2;
        let y = m * w2 * (1.0 - s2);
        Vec3::new(x, y, 0.0)
    }
}

impl Default for YangCao {
    fn default() -> Self { Self::new() }
}

impl Attractor for YangCao {
    fn step(&mut self) -> Option<Point> {
        let prev_state = self.state;
        let plot = self.map_step();
        if !self.state.is_finite() || !plot.is_finite() {
            self.state = prev_state;
            return None;
        }
        let speed = (plot - self.last_plot).length();
        self.last_plot = plot;
        Some(Point { pos: plot, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(0.10, 0.10, 0.0);
        self.last_plot = Vec3::ZERO;
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
