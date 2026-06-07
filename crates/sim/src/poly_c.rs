use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params: [p0..p17]  (18 total)
// Equations (discrete map — cyclic structure across x,y,z):
//   x' = p0  + x*(p1  + p2*x  + p3*y)  + y*(p4  + p5*y)
//   y' = p6  + y*(p7  + p8*y  + p9*z)  + z*(p10 + p11*z)
//   z' = p12 + z*(p13 + p14*z + p15*x) + x*(p16 + p17*x)

const DESCS: &[ParamDesc] = &[
    // X equation
    ParamDesc { name: "P0",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.935 },
    ParamDesc { name: "P1",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.282 },
    ParamDesc { name: "P2",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.003 },
    ParamDesc { name: "P3",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.772 },
    ParamDesc { name: "P4",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.111 },
    ParamDesc { name: "P5",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.102 },
    // Y equation
    ParamDesc { name: "P6",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.368 },
    ParamDesc { name: "P7",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.473 },
    ParamDesc { name: "P8",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.811 },
    ParamDesc { name: "P9",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -1.111 },
    ParamDesc { name: "P10", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.748 },
    ParamDesc { name: "P11", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.681 },
    // Z equation
    ParamDesc { name: "P12", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.349 },
    ParamDesc { name: "P13", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.971 },
    ParamDesc { name: "P14", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.766 },
    ParamDesc { name: "P15", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.144 },
    ParamDesc { name: "P16", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.245 },
    ParamDesc { name: "P17", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.129 },
];

pub struct PolyC {
    state:  Vec3,
    params: [f32; 18],
}

impl PolyC {
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
        self.state.x = p[0]  + x * (p[1]  + p[2]  * x + p[3]  * y) + y * (p[4]  + p[5]  * y);
        self.state.y = p[6]  + y * (p[7]  + p[8]  * y + p[9]  * z) + z * (p[10] + p[11] * z);
        self.state.z = p[12] + z * (p[13] + p[14] * z + p[15] * x) + x * (p[16] + p[17] * x);
    }
}

impl Default for PolyC {
    fn default() -> Self { Self::new() }
}

impl Attractor for PolyC {
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
