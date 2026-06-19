use crate::attractor::ParamDesc;
use crate::lorenz::Lorenz;
use crate::lorenz84::Lorenz84;
use crate::rossler::Rossler;
use crate::thomas::Thomas;
use crate::chaotic_flow::ChaoticFlow;
use crate::clifford::Clifford;
use crate::icon::Icon;
use crate::icon_b::IconB;
use crate::pickover::Pickover;
use crate::poly_a::PolyA;
use crate::poly_abs::PolyAbs;
use crate::poly_b::PolyB;
use crate::poly_c::PolyC;
use crate::poly_pow::PolyPow;
use crate::poly_sin::PolySin;
use crate::poly_sprott::PolySprott;
use crate::attractor::Attractor;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, Serialize, Deserialize)]
pub enum AttractorType {
    #[default]
    Lorenz,
    Lorenz84,
    Rossler,
    Thomas,
    ChaoticFlow,
    Pickover,
    Clifford,
    Icon,
    IconB,
    PolyA,
    PolyAbs,
    PolyB,
    PolyC,
    PolyPow,
    PolySin,
    PolySprott,
}

impl AttractorType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Lorenz      => "Lorenz",
            Self::Lorenz84    => "Lorenz 84",
            Self::Rossler     => "Rössler",
            Self::Thomas      => "Thomas",
            Self::ChaoticFlow => "Chaotic Flow",
            Self::Pickover    => "Pickover",
            Self::Clifford    => "Clifford",
            Self::Icon        => "Icon",
            Self::IconB       => "Icon B",
            Self::PolyA       => "Polynomial A",
            Self::PolyAbs     => "Polynomial Abs",
            Self::PolyB       => "Polynomial B",
            Self::PolyC       => "Polynomial C",
            Self::PolyPow     => "Polynomial Power",
            Self::PolySin     => "Polynomial Sin",
            Self::PolySprott  => "Polynomial Sprott",
        }
    }

    pub fn descriptors(self) -> &'static [ParamDesc] {
        match self {
            Self::Lorenz      => Lorenz::param_descriptors(),
            Self::Lorenz84    => Lorenz84::param_descriptors(),
            Self::Rossler     => Rossler::param_descriptors(),
            Self::Thomas      => Thomas::param_descriptors(),
            Self::ChaoticFlow => ChaoticFlow::param_descriptors(),
            Self::Pickover    => Pickover::param_descriptors(),
            Self::Clifford    => Clifford::param_descriptors(),
            Self::Icon        => Icon::param_descriptors(),
            Self::IconB       => IconB::param_descriptors(),
            Self::PolyA       => PolyA::param_descriptors(),
            Self::PolyAbs     => PolyAbs::param_descriptors(),
            Self::PolyB       => PolyB::param_descriptors(),
            Self::PolyC       => PolyC::param_descriptors(),
            Self::PolyPow     => PolyPow::param_descriptors(),
            Self::PolySin     => PolySin::param_descriptors(),
            Self::PolySprott  => PolySprott::param_descriptors(),
        }
    }

    /// Returns true for continuous-time ODE attractors integrated with Euler steps.
    /// These have one Lyapunov exponent ≈ 0 along the flow, so D_KY = 2 + λ₁/|λ₃|
    /// rather than the map formula D_KY = 1 + λ₁/|λ₂|.
    pub fn is_flow(self) -> bool {
        matches!(self,
            Self::Lorenz | Self::Lorenz84 | Self::Rossler | Self::Thomas
            // ChaoticFlow excluded: at its working dT range (~0.5–1.5) the Euler
            // discretization behaves as a 3D discrete map, not a continuous flow.
            // The map D_KY formula (1 + λ₁/|λ₂|) applies.
        )
    }

    pub const ALL: &'static [Self] = &[
        Self::Lorenz,
        Self::Lorenz84,
        Self::Rossler,
        Self::Thomas,
        Self::ChaoticFlow,
        Self::Pickover,
        Self::Clifford,
        Self::Icon,
        Self::IconB,
        Self::PolyA,
        Self::PolyAbs,
        Self::PolyB,
        Self::PolyC,
        Self::PolyPow,
        Self::PolySin,
        Self::PolySprott,
    ];
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AttractorConfig {
    pub attractor_type: AttractorType,
    pub params:         Vec<f32>,
    /// Per-parameter freeze flags: search skips frozen indices and keeps the
    /// current value instead of sampling a new one.
    pub frozen:         Vec<bool>,
}

impl AttractorConfig {
    pub fn new(t: AttractorType) -> Self {
        let descs = t.descriptors();
        Self {
            attractor_type: t,
            params: descs.iter().map(|d| d.default).collect(),
            frozen: vec![false; descs.len()],
        }
    }

    pub fn descriptors(&self) -> &'static [ParamDesc] {
        self.attractor_type.descriptors()
    }

    /// Run a short CPU trajectory to estimate the attractor's bounding box.
    /// Used to fit the arcball camera target and distance after a type switch.
    pub fn estimate_bounds_cpu(&self) -> (glam::Vec3, glam::Vec3) {
        use crate::attractor::estimate_bounds;
        match self.attractor_type {
            AttractorType::Lorenz => {
                let mut a = Lorenz::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Lorenz84 => {
                let mut a = Lorenz84::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Rossler => {
                let mut a = Rossler::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Thomas => {
                let mut a = Thomas::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ChaoticFlow => {
                let mut a = ChaoticFlow::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Pickover => {
                let mut a = Pickover::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Clifford => {
                let mut a = Clifford::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Icon => {
                let mut a = Icon::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::IconB => {
                let mut a = IconB::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::PolyA => {
                let mut a = PolyA::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::PolyAbs => {
                let mut a = PolyAbs::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::PolyB => {
                let mut a = PolyB::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::PolyC => {
                let mut a = PolyC::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::PolyPow => {
                let mut a = PolyPow::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::PolySin => {
                let mut a = PolySin::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::PolySprott => {
                let mut a = PolySprott::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
        }
    }
}
