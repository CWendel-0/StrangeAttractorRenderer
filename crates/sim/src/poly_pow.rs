use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Discrete map (iterated, no dt).
// x' = P0  + P1·x  + P2·y  + P3·z  + P4·|x|  + P5·|y|  + P6·|z|^P7
// y' = P8  + P9·x  + P10·y + P11·z + P12·|x| + P13·|y| + P14·|z|^P15
// z' = P16 + P17·x + P18·y + P19·z + P20·|x| + P21·|y| + P22·|z|^P23

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "P0",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.273  },
    ParamDesc { name: "P1",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.989  },
    ParamDesc { name: "P2",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.872  },
    ParamDesc { name: "P3",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.525  },
    ParamDesc { name: "P4",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.342  },
    ParamDesc { name: "P5",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.758  },
    ParamDesc { name: "P6",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.872  },
    ParamDesc { name: "P7",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.579  },
    ParamDesc { name: "P8",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.613  },
    ParamDesc { name: "P9",  kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.04   },
    ParamDesc { name: "P10", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.046  },
    ParamDesc { name: "P11", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.58   },
    ParamDesc { name: "P12", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.021  },
    ParamDesc { name: "P13", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.071  },
    ParamDesc { name: "P14", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.073  },
    ParamDesc { name: "P15", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.411  },
    ParamDesc { name: "P16", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.641  },
    ParamDesc { name: "P17", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.857  },
    ParamDesc { name: "P18", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.075  },
    ParamDesc { name: "P19", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.461  },
    ParamDesc { name: "P20", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.57   },
    ParamDesc { name: "P21", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default: -0.603  },
    ParamDesc { name: "P22", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  1.042  },
    ParamDesc { name: "P23", kind: ParamKind::Continuous, min: -1.2, max: 1.2, default:  0.54   },
];

pub struct PolyPow {
    state:  Vec3,
    params: [f32; 24],
}

impl PolyPow {
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
        self.state.x = p[0]  + p[1]*x  + p[2]*y  + p[3]*z  + p[4]*x.abs()  + p[5]*y.abs()  + p[6]*z.abs().powf(p[7]);
        self.state.y = p[8]  + p[9]*x  + p[10]*y + p[11]*z + p[12]*x.abs() + p[13]*y.abs() + p[14]*z.abs().powf(p[15]);
        self.state.z = p[16] + p[17]*x + p[18]*y + p[19]*z + p[20]*x.abs() + p[21]*y.abs() + p[22]*z.abs().powf(p[23]);
    }
}

impl Default for PolyPow {
    fn default() -> Self { Self::new() }
}

impl Attractor for PolyPow {
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
