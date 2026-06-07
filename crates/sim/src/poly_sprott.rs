use glam::Vec3;
use crate::attractor::{Attractor, ParamDesc, ParamKind, Point};

// Sprott 3D polynomial discrete map.
// Order O (2-5) selects number of monomials per equation: C(O+3,3) = 10/20/35/56.
//
// Monomial ordering matches Sprott's book (Chaoscope-compatible):
//   [0]=1  [1]=x  [2]=x²  [3]=xy  [4]=xz  [5]=y  [6]=y²  [7]=yz  [8]=z  [9]=z²
//   deg 3:  x³, x²y, x²z, xy², xyz, xz², y³, y²z, yz², z³
//   deg 4:  x⁴, x³y, x³z, x²y², x²yz, x²z², xy³, xy²z, xyz², xz³, y⁴, y³z, y²z², yz³, z⁴
//   deg 5:  x⁵, x⁴y, x⁴z, x³y², x³yz, x³z², x²y³, x²y²z, x²yz², x²z³,
//           xy⁴, xy³z, xy²z², xyz³, xz⁴, y⁵, y⁴z, y³z², y²z³, yz⁴, z⁵
//
// P0       = order (integer 2-5)
// P1-P56   = X equation coefficients (slots 56; high-order unused terms are 0)
// P57-P112 = Y equation coefficients
// P113-P168 = Z equation coefficients

const X_DEF: [f32; 10] = [
    -0.554, -0.600,  0.226, -0.559,  0.255,
     0.736, -0.071, -0.598,  1.108, -1.182,
];
const Y_DEF: [f32; 10] = [
     0.504, -0.884, -0.528, -0.709,  0.500,
    -0.906, -0.416,  0.532, -0.519,  0.485,
];
const Z_DEF: [f32; 10] = [
     0.557, -0.020,  0.081, -0.415,  0.074,
    -0.233, -0.196, -0.177, -1.107,  0.511,
];

fn eval_poly(c: &[f32], x: f32, y: f32, z: f32, order: usize) -> f32 {
    let x2 = x * x; let y2 = y * y; let z2 = z * z;
    // Sprott ordering: 1, x, x², xy, xz, y, y², yz, z, z²
    let mut r = c[0] + c[1]*x + c[2]*x2 + c[3]*(x*y) + c[4]*(x*z)
              + c[5]*y + c[6]*y2 + c[7]*(y*z)
              + c[8]*z + c[9]*z2;
    if order >= 3 {
        let x3 = x2*x; let y3 = y2*y; let z3 = z2*z;
        r += c[10]*x3  + c[11]*(x2*y) + c[12]*(x2*z) + c[13]*(x*y2)
           + c[14]*(x*y*z) + c[15]*(x*z2) + c[16]*y3 + c[17]*(y2*z)
           + c[18]*(y*z2) + c[19]*z3;
        if order >= 4 {
            let x4 = x3*x; let y4 = y3*y; let z4 = z3*z;
            r += c[20]*x4  + c[21]*(x3*y) + c[22]*(x3*z) + c[23]*(x2*y2)
               + c[24]*(x2*y*z) + c[25]*(x2*z2) + c[26]*(x*y3) + c[27]*(x*y2*z)
               + c[28]*(x*y*z2) + c[29]*(x*z3) + c[30]*y4 + c[31]*(y3*z)
               + c[32]*(y2*z2) + c[33]*(y*z3) + c[34]*z4;
            if order >= 5 {
                let x5 = x4*x; let y5 = y4*y; let z5 = z4*z;
                r += c[35]*x5   + c[36]*(x4*y)  + c[37]*(x4*z)  + c[38]*(x3*y2)
                   + c[39]*(x3*y*z) + c[40]*(x3*z2) + c[41]*(x2*y3) + c[42]*(x2*y2*z)
                   + c[43]*(x2*y*z2) + c[44]*(x2*z3) + c[45]*(x*y4) + c[46]*(x*y3*z)
                   + c[47]*(x*y2*z2) + c[48]*(x*y*z3) + c[49]*(x*z4) + c[50]*y5
                   + c[51]*(y4*z) + c[52]*(y3*z2) + c[53]*(y2*z3) + c[54]*(y*z4)
                   + c[55]*z5;
            }
        }
    }
    r
}

pub struct PolySprott {
    state:  Vec3,
    params: [f32; 169],
}

impl PolySprott {
    pub fn new() -> Self {
        let mut s = Self {
            state:  Vec3::new(0.1, 0.1, 0.0),
            params: [0.0; 169],
        };
        s.params[0] = 2.0;
        s.params[1..=10].copy_from_slice(&X_DEF);
        s.params[57..=66].copy_from_slice(&Y_DEF);
        s.params[113..=122].copy_from_slice(&Z_DEF);
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
        let order = self.params[0].round() as usize;
        let (x, y, z) = (self.state.x, self.state.y, self.state.z);
        self.state.x = eval_poly(&self.params[1..57],   x, y, z, order);
        self.state.y = eval_poly(&self.params[57..113],  x, y, z, order);
        self.state.z = eval_poly(&self.params[113..169], x, y, z, order);
    }
}

impl Default for PolySprott {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attractor::Attractor;

    #[test]
    fn default_trajectory_stays_bounded() {
        let mut a = PolySprott::new();
        let mut valid = 0usize;
        for _ in 0..10_000 {
            match a.step() {
                Some(p) => {
                    assert!(p.pos.is_finite(), "trajectory diverged: {:?}", p.pos);
                    assert!(p.pos.x.abs() < 100.0 && p.pos.y.abs() < 100.0 && p.pos.z.abs() < 100.0,
                            "trajectory escaped: {:?}", p.pos);
                    valid += 1;
                }
                None => { panic!("step() returned None — trajectory diverged"); }
            }
        }
        assert!(valid > 9_000, "too many diverged steps: only {} valid", valid);
    }
}

impl Attractor for PolySprott {
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

    fn param_descriptors() -> &'static [ParamDesc] {
        static DESCS: std::sync::OnceLock<Box<[ParamDesc]>> = std::sync::OnceLock::new();
        DESCS.get_or_init(|| {
            let mut v: Vec<ParamDesc> = Vec::with_capacity(169);
            v.push(ParamDesc { name: "Order", kind: ParamKind::Integer, min: 2.0, max: 5.0, default: 2.0 });
            for i in 1usize..=168 {
                let default = if (1..=10).contains(&i)     { X_DEF[i - 1]   }
                         else if (57..=66).contains(&i)    { Y_DEF[i - 57]  }
                         else if (113..=122).contains(&i)  { Z_DEF[i - 113] }
                         else                              { 0.0 };
                let name: &'static str = Box::leak(format!("P{i}").into_boxed_str());
                v.push(ParamDesc { name, kind: ParamKind::Continuous, min: -1.5, max: 1.5, default });
            }
            v.into_boxed_slice()
        }).as_ref()
    }

    fn params(&self) -> &[f32] { &self.params }

    fn pos(&self) -> Vec3 { self.state }
    fn set_pos(&mut self, pos: Vec3) { self.state = pos; }
}
