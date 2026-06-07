use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Discrete map (iterated, no dt).
// x' = P0  + P1·x  + P2·y  + P3·z  + P4·|x|  + P5·|y|  + P6·|z|
// y' = P7  + P8·x  + P9·y  + P10·z + P11·|x| + P12·|y| + P13·|z|
// z' = P14 + P15·x + P16·y + P17·z + P18·|x| + P19·|y| + P20·|z|

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "P0",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.211 },
    ParamDesc { name: "P1",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.483 },
    ParamDesc { name: "P2",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.25  },
    ParamDesc { name: "P3",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.191 },
    ParamDesc { name: "P4",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.48  },
    ParamDesc { name: "P5",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.948 },
    ParamDesc { name: "P6",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.394 },
    ParamDesc { name: "P7",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.983 },
    ParamDesc { name: "P8",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.193 },
    ParamDesc { name: "P9",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.287 },
    ParamDesc { name: "P10", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.473 },
    ParamDesc { name: "P11", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.317 },
    ParamDesc { name: "P12", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.764 },
    ParamDesc { name: "P13", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.366 },
    ParamDesc { name: "P14", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.673 },
    ParamDesc { name: "P15", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.691 },
    ParamDesc { name: "P16", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -1.082 },
    ParamDesc { name: "P17", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.665 },
    ParamDesc { name: "P18", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.931 },
    ParamDesc { name: "P19", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.042 },
    ParamDesc { name: "P20", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.653 },
];

pub struct PolyAbs {
    state:  Vec3,
    params: [f32; 21],
}

impl PolyAbs {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.0),
            params: std::array::from_fn(|i| DESCS[i].default),
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..500 {
            self.map_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.1, 0.1, 0.0);
                break;
            }
        }
    }

    fn map_step(&mut self) {
        let p = &self.params;
        let (x, y, z) = (self.state.x, self.state.y, self.state.z);
        self.state.x = p[0]  + p[1]*x  + p[2]*y  + p[3]*z  + p[4]*x.abs()  + p[5]*y.abs()  + p[6]*z.abs();
        self.state.y = p[7]  + p[8]*x  + p[9]*y  + p[10]*z + p[11]*x.abs() + p[12]*y.abs() + p[13]*z.abs();
        self.state.z = p[14] + p[15]*x + p[16]*y + p[17]*z + p[18]*x.abs() + p[19]*y.abs() + p[20]*z.abs();
    }
}

impl Default for PolyAbs {
    fn default() -> Self { Self::new() }
}

impl Attractor for PolyAbs {
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
        self.state = Vec3::new(0.1, 0.1, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }
    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
