use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Discrete map (iterated, no dt).
// x' = P0  + P1·x  + P2·y  + P3·z  + P4·sin(P5·x+P6)   + P7·sin(P8·y+P9)   + P10·sin(P11·z+P12)
// y' = P13 + P14·x + P15·y + P16·z + P17·sin(P18·x+P19) + P20·sin(P21·y+P22) + P23·sin(P24·z+P25)
// z' = P26 + P27·x + P28·y + P29·z + P30·sin(P31·x+P32) + P33·sin(P34·y+P35) + P36·sin(P37·z+P38)

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "P0",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.229 },
    ParamDesc { name: "P1",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.042 },
    ParamDesc { name: "P2",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.619 },
    ParamDesc { name: "P3",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.522 },
    ParamDesc { name: "P4",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.758 },
    ParamDesc { name: "P5",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.196 },
    ParamDesc { name: "P6",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.441 },
    ParamDesc { name: "P7",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.102 },
    ParamDesc { name: "P8",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.947 },
    ParamDesc { name: "P9",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.134 },
    ParamDesc { name: "P10", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.485 },
    ParamDesc { name: "P11", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.152 },
    ParamDesc { name: "P12", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.057 },
    ParamDesc { name: "P13", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.152 },
    ParamDesc { name: "P14", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.217 },
    ParamDesc { name: "P15", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.643 },
    ParamDesc { name: "P16", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.862 },
    ParamDesc { name: "P17", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.112 },
    ParamDesc { name: "P18", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.6   },
    ParamDesc { name: "P19", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.061 },
    ParamDesc { name: "P20", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.452 },
    ParamDesc { name: "P21", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.757 },
    ParamDesc { name: "P22", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.344 },
    ParamDesc { name: "P23", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.086 },
    ParamDesc { name: "P24", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.739 },
    ParamDesc { name: "P25", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.45  },
    ParamDesc { name: "P26", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.797 },
    ParamDesc { name: "P27", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.113 },
    ParamDesc { name: "P28", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.978 },
    ParamDesc { name: "P29", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.36  },
    ParamDesc { name: "P30", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -1.183 },
    ParamDesc { name: "P31", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.656 },
    ParamDesc { name: "P32", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.071 },
    ParamDesc { name: "P33", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.56  },
    ParamDesc { name: "P34", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.533 },
    ParamDesc { name: "P35", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.392 },
    ParamDesc { name: "P36", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.028 },
    ParamDesc { name: "P37", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.93  },
    ParamDesc { name: "P38", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.358 },
];

pub struct PolySin {
    state:  Vec3,
    params: [f32; 39],
}

impl PolySin {
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
        self.state.x = p[0]  + p[1]*x  + p[2]*y  + p[3]*z
                     + p[4]  * (p[5]*x  + p[6]).sin()
                     + p[7]  * (p[8]*y  + p[9]).sin()
                     + p[10] * (p[11]*z + p[12]).sin();
        self.state.y = p[13] + p[14]*x + p[15]*y + p[16]*z
                     + p[17] * (p[18]*x + p[19]).sin()
                     + p[20] * (p[21]*y + p[22]).sin()
                     + p[23] * (p[24]*z + p[25]).sin();
        self.state.z = p[26] + p[27]*x + p[28]*y + p[29]*z
                     + p[30] * (p[31]*x + p[32]).sin()
                     + p[33] * (p[34]*y + p[35]).sin()
                     + p[36] * (p[37]*z + p[38]).sin();
    }
}

impl Default for PolySin {
    fn default() -> Self { Self::new() }
}

impl Attractor for PolySin {
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
