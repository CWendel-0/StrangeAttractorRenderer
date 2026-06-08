use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Params layout (22 total):
//   [0..11]  = m0..m11    (float multipliers)
//   [12..14] = Op0,Op1,Op2  (for X eq: m0,m1,m2)
//   [15..17] = Op4,Op5,Op6  (for Y eq: m4,m5,m6)
//   [18..20] = Op8,Op9,Op10 (for Z eq: m8,m9,m10)
//   [21]     = dt
//
// Equations (Euler):
//   dx = m0·v(Op0) + m1·v(Op1) + m2·v(Op2) + m3        ← m3 always constant
//   dy = m4·v(Op4) + m5·v(Op5) + m6·v(Op6) + m7        ← m7 always constant
//   dz = m8·v(Op8) + m9·v(Op9) + m10·v(Op10) + m11     ← m11 always constant
//   x' = x + dt·dx   (matches Chaoscope Chaotic Flow)
//
// Defaults: Chaoscope example that produces a visible torus-like attractor.

const DESCS: &[ParamDesc] = &[
    ParamDesc { name: "m0",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: -0.863 },
    ParamDesc { name: "m1",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: -0.244 },
    ParamDesc { name: "m2",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: -0.657 },
    ParamDesc { name: "m3",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: -0.643 },
    ParamDesc { name: "m4",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: -0.292 },
    ParamDesc { name: "m5",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default:  0.179 },
    ParamDesc { name: "m6",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: -0.390 },
    ParamDesc { name: "m7",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: -0.311 },
    ParamDesc { name: "m8",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default:  0.917 },
    ParamDesc { name: "m9",  kind: ParamKind::Continuous, min: -1.0, max: 1.0, default:  0.824 },
    ParamDesc { name: "m10", kind: ParamKind::Continuous, min: -1.0, max: 1.0, default:  0.198 },
    ParamDesc { name: "m11", kind: ParamKind::Continuous, min: -1.0, max: 1.0, default: -0.436 },
    // Ops for X equation (Op for m0, m1, m2; m3 is always constant)
    ParamDesc { name: "Op0", kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 0.0 },
    ParamDesc { name: "Op1", kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 0.0 },
    ParamDesc { name: "Op2", kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 1.0 },
    // Ops for Y equation (Op for m4, m5, m6; m7 is always constant)
    ParamDesc { name: "Op4", kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 3.0 },
    ParamDesc { name: "Op5", kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 2.0 },
    ParamDesc { name: "Op6", kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 1.0 },
    // Ops for Z equation (Op for m8, m9, m10; m11 is always constant)
    ParamDesc { name: "Op8",  kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 1.0 },
    ParamDesc { name: "Op9",  kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 0.0 },
    ParamDesc { name: "Op10", kind: ParamKind::Enum(&["-","X","Y","Z"]), min: 0.0, max: 3.0, default: 1.0 },
    ParamDesc { name: "dT",   kind: ParamKind::Continuous, min: 0.3, max: 1.5, default: 0.945 },
];

#[inline]
fn eval_v(op: u8, x: f32, y: f32, z: f32) -> f32 {
    match op {
        1 => x,
        2 => y,
        3 => z,
        _ => 1.0, // "-" / Const → multiply coefficient by 1
    }
}

pub struct ChaoticFlow {
    state:  Vec3,
    params: [f32; 22],
}

impl ChaoticFlow {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.0, 0.1, 0.0),
            params: std::array::from_fn(|i| DESCS[i].default),
        };
        s.warm_up();
        s
    }

    fn warm_up(&mut self) {
        for _ in 0..1000 {
            self.euler_step();
            if !self.state.is_finite() {
                self.state = Vec3::new(0.0, 0.1, 0.0);
                break;
            }
        }
    }

    fn euler_step(&mut self) {
        let p = &self.params;
        let (x, y, z) = (self.state.x, self.state.y, self.state.z);
        let dt = p[21];
        let op = |i: usize| p[12 + i] as u8;

        // Matrix-vector form: each column multiplies [x,y,z]; Op adds a quadratic factor.
        // dx = m0·f(Op0)·x + m1·f(Op1)·y + m2·f(Op2)·z + m3
        let dx = p[0]*eval_v(op(0),x,y,z)*x + p[1]*eval_v(op(1),x,y,z)*y
               + p[2]*eval_v(op(2),x,y,z)*z + p[3];
        let dy = p[4]*eval_v(op(3),x,y,z)*x + p[5]*eval_v(op(4),x,y,z)*y
               + p[6]*eval_v(op(5),x,y,z)*z + p[7];
        let dz = p[8]*eval_v(op(6),x,y,z)*x + p[9]*eval_v(op(7),x,y,z)*y
               + p[10]*eval_v(op(8),x,y,z)*z + p[11];

        self.state.x = x + dt * dx;
        self.state.y = y + dt * dy;
        self.state.z = z + dt * dz;
    }
}

impl Default for ChaoticFlow {
    fn default() -> Self { Self::new() }
}

impl Attractor for ChaoticFlow {
    fn step(&mut self) -> Option<Point> {
        let prev = self.state;
        self.euler_step();
        if !self.state.is_finite() {
            self.state = prev;
            return None;
        }
        let speed = (self.state - prev).length() / self.params[21].max(1e-9);
        Some(Point { pos: self.state, speed })
    }

    fn reset(&mut self, params: &[f32]) {
        for (dst, src) in self.params.iter_mut().zip(params.iter()) {
            *dst = *src;
        }
        self.state = Vec3::new(0.0, 0.1, 0.0);
        self.warm_up();
    }

    fn param_descriptors() -> &'static [ParamDesc] { DESCS }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
