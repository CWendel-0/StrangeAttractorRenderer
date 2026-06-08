use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use crossbeam_channel::{bounded, Receiver};
use glam::Vec3;

use crate::attractor::Attractor;
use crate::config::{AttractorConfig, AttractorType};
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

pub struct SearchResult {
    pub attractor_type: AttractorType,
    pub params:         Vec<f32>,
    pub bb_min:         Vec3,
    pub bb_max:         Vec3,
}

pub struct SearchWorker {
    pub result_rx: Receiver<SearchResult>,
    stop:          Arc<AtomicBool>,
    handles:       Vec<JoinHandle<()>>,
}

impl SearchWorker {
    /// Spawn parallel background threads that race to find a chaotic attractor
    /// of the same type as `config`.  The first thread to find one signals all
    /// others to stop via the shared stop flag, then sends the result.
    pub fn spawn(config: AttractorConfig) -> Self {
        let (result_tx, result_rx) = bounded::<SearchResult>(1);
        let stop = Arc::new(AtomicBool::new(false));

        // Leave at least one core for the GPU/render thread; cap at 4.
        let n_threads = thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1).min(4))
            .unwrap_or(2);

        let handles = (0..n_threads)
            .map(|_| {
                let result_tx = result_tx.clone();
                let stop_flag = Arc::clone(&stop);
                let atype       = config.attractor_type;
                let descs       = atype.descriptors();
                let params_init = config.params.clone();
                let frozen      = config.frozen.clone();

                thread::spawn(move || {
                    let mut rng = Lcg64::seeded();

                    loop {
                        if stop_flag.load(Ordering::Relaxed) { break; }

                        let params = sample_params(descs, &params_init, &frozen, &mut rng);

                        if let Some((bb_min, bb_max)) =
                            test_params(atype, &params, &stop_flag)
                        {
                            // Signal all threads to stop; only one try_send will
                            // succeed (channel capacity 1).
                            stop_flag.store(true, Ordering::Relaxed);
                            let _ = result_tx.try_send(SearchResult {
                                attractor_type: atype,
                                params,
                                bb_min,
                                bb_max,
                            });
                            break;
                        }
                    }
                })
            })
            .collect();

        SearchWorker { result_rx, stop, handles }
    }
}

impl Drop for SearchWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

// ---- Simple LCG RNG (no external crate) -----------------------------------

struct Lcg64 { state: u64 }

impl Lcg64 {
    fn seeded() -> Self {
        use std::time::SystemTime;
        // Mix thread ID into seed so parallel workers explore different regions.
        let tid = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            std::thread::current().id().hash(&mut h);
            h.finish()
        };
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ d.as_secs().wrapping_mul(0x9e3779b97f4a7c15))
            .unwrap_or(0xcafe_babe_dead_beef);
        let mut s = t ^ tid;
        s ^= s >> 30;
        s = s.wrapping_mul(0xbf58476d1ce4e5b9);
        s ^= s >> 27;
        s = s.wrapping_mul(0x94d049bb133111eb);
        s ^= s >> 31;
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn f32_01(&mut self) -> f32 {
        // >> 33 yields 31 bits; divide by 2³¹ to get [0, 1).
        (self.next_u64() >> 33) as f32 * (1.0 / 2_147_483_648.0_f32)
    }

    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32_01() * (hi - lo)
    }

    fn range_int(&mut self, lo: f32, hi: f32) -> f32 {
        let n = (hi - lo + 1.0).round().max(1.0) as u64;
        (self.next_u64() % n) as f32 + lo
    }
}

// ---- Parameter sampling ----------------------------------------------------

fn sample_params(
    descs:   &[crate::attractor::ParamDesc],
    current: &[f32],
    frozen:  &[bool],
    rng:     &mut Lcg64,
) -> Vec<f32> {
    use crate::attractor::ParamKind;
    descs.iter().enumerate().map(|(i, desc)| {
        if frozen.get(i).copied().unwrap_or(false) {
            current.get(i).copied().unwrap_or(desc.default)
        } else {
            match &desc.kind {
                ParamKind::Continuous    => rng.range_f32(desc.min, desc.max),
                ParamKind::Integer       => rng.range_int(desc.min, desc.max),
                ParamKind::Enum(choices) => rng.range_int(0.0, choices.len() as f32 - 1.0),
            }
        }
    }).collect()
}

// ---- Chaoticity test -------------------------------------------------------

fn test_params(t: AttractorType, params: &[f32], stop: &AtomicBool) -> Option<(Vec3, Vec3)> {
    let is_flow = t.is_flow();
    macro_rules! probe {
        ($T:ty) => {{
            let mut main    = <$T>::new(); main.reset(params);
            let mut shadow1 = <$T>::new(); shadow1.reset(params);
            let mut shadow2 = <$T>::new(); shadow2.reset(params);
            let mut shadow3 = <$T>::new(); shadow3.reset(params);
            run_test(&mut main, &mut shadow1, &mut shadow2, &mut shadow3, is_flow, stop)
        }};
    }
    match t {
        AttractorType::Lorenz      => probe!(Lorenz),
        AttractorType::Lorenz84    => probe!(Lorenz84),
        AttractorType::Rossler     => probe!(Rossler),
        AttractorType::Thomas      => probe!(Thomas),
        AttractorType::ChaoticFlow => probe!(ChaoticFlow),
        AttractorType::Clifford    => probe!(Clifford),
        AttractorType::Icon        => probe!(Icon),
        AttractorType::IconB       => probe!(IconB),
        AttractorType::Pickover    => probe!(Pickover),
        AttractorType::PolyA       => probe!(PolyA),
        AttractorType::PolyAbs     => probe!(PolyAbs),
        AttractorType::PolyB       => probe!(PolyB),
        AttractorType::PolyC       => probe!(PolyC),
        AttractorType::PolyPow     => probe!(PolyPow),
        AttractorType::PolySin     => probe!(PolySin),
        AttractorType::PolySprott  => probe!(PolySprott),
    }
}

/// Shadow-orbit Lyapunov + Kaplan-Yorke dimension search (Sprott §3.3 / §3.4).
///
/// Four trajectories run in parallel: `main`, `shadow1`, `shadow2`, `shadow3`.
/// The three shadows are kept mutually orthogonal via Gram-Schmidt after each
/// step, giving the three Lyapunov exponents λ₁ ≥ λ₂ ≥ λ₃.
///
/// **Discrete-map attractors** (`is_flow = false`):
///   D_KY = 1 + λ₁/|λ₂|   when λ₂ < 0  →  D ∈ (1, 2)
///   Filter: D_KY ∈ [1.05, 1.9],  λ₁ ∈ [0.03, 0.3]
///
/// **ODE-flow attractors** (`is_flow = true`):
///   One exponent is always ≈ 0 along the flow direction (λ₂ ≈ 0), so the
///   correct Kaplan-Yorke formula uses the third exponent:
///   D_KY = 2 + λ₁/|λ₃|   when λ₃ < 0  →  D ∈ (2, 3)
///   A true limit cycle gives D_KY = 2.0 exactly; strange attractors have D > 2.
///   Filter: D_KY ∈ [2.01, 2.9],  λ₁ > 0
fn run_test<A: Attractor>(
    main:    &mut A,
    shadow1: &mut A,
    shadow2: &mut A,
    shadow3: &mut A,
    is_flow: bool,
    stop:    &AtomicBool,
) -> Option<(Vec3, Vec3)> {
    const WARMUP:          usize = 200;
    const TEST:            usize = 8_000;
    const EARLY_CHK:       usize = 500;
    const LY_EARLY_THRESH: f32   = -0.1;
    const LY_EPS:          f32   = 1e-4;
    // Map thresholds
    const LY_MIN:          f32   = 0.03;
    const LY_MAX:          f32   = 0.3;
    const DIM_MIN:         f32   = 1.5;
    const DIM_MAX:         f32   = 10.0;
    // Flow thresholds (D_KY = 2 + λ₁/|λ₃|; limit cycles give exactly 2.0)
    const FLOW_DIM_MIN:    f32   = 2.05;
    const FLOW_DIM_MAX:    f32   = 10.0;
    const BOUND:           f32   = 1e6;
    const CHECK_INT:       usize = 500;

    // --- Warmup: let the main trajectory settle onto the attractor ----------
    let mut fixed_streak = 0usize;
    let mut prev = main.pos();

    for i in 0..WARMUP {
        if i % CHECK_INT == 0 && stop.load(Ordering::Relaxed) { return None; }
        match main.step() {
            None => return None,
            Some(pt) => {
                if !pt.pos.is_finite() || pt.pos.length() > BOUND { return None; }
                let disp = (pt.pos - prev).length();
                if disp < 1e-7 {
                    fixed_streak += 1;
                    if fixed_streak > 5 { return None; }
                } else {
                    fixed_streak = 0;
                }
                prev = pt.pos;
            }
        }
    }

    // --- Place shadows at main + ε along three orthogonal axes --------------
    shadow1.set_pos(main.pos() + Vec3::X * LY_EPS);
    shadow2.set_pos(main.pos() + Vec3::Y * LY_EPS);
    shadow3.set_pos(main.pos() + Vec3::Z * LY_EPS);

    let mut d1_hat = Vec3::X;
    let mut d2_hat = Vec3::Y;

    // --- Shadow warmup: align directions before accumulating ----------------
    // 10,000 steps ensures shadows converge to the true expanding directions
    // even for slowly-contracting limit cycles at very small dT values.
for _ in 0..10_000 {
        let pt_m = match main.step() {
            None => return None,
            Some(pt) => pt,
        };
        if !pt_m.pos.is_finite() || pt_m.pos.length() > BOUND { return None; }

        if let Some(pt_s1) = shadow1.step() {
            let sep1 = pt_s1.pos - pt_m.pos;
            let len1 = sep1.length();
            if len1 > 0.0 && len1.is_finite() {
                d1_hat = sep1 / len1;
                shadow1.set_pos(pt_m.pos + d1_hat * LY_EPS);
            }
        }
        if let Some(pt_s2) = shadow2.step() {
            let sep2      = pt_s2.pos - pt_m.pos;
            let sep2_orth = sep2 - sep2.dot(d1_hat) * d1_hat;
            let len2      = sep2_orth.length();
            if len2 > 1e-30 && len2.is_finite() {
                d2_hat = sep2_orth / len2;
                shadow2.set_pos(pt_m.pos + d2_hat * LY_EPS);
            } else if len2.is_finite() {
                let perp = if d1_hat.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                d2_hat = (perp - perp.dot(d1_hat) * d1_hat).normalize();
                shadow2.set_pos(pt_m.pos + d2_hat * LY_EPS);
            }
        }
        if is_flow {
            if let Some(pt_s3) = shadow3.step() {
                let sep3      = pt_s3.pos - pt_m.pos;
                let sep3_orth = sep3
                    - sep3.dot(d1_hat) * d1_hat
                    - sep3.dot(d2_hat) * d2_hat;
                let len3 = sep3_orth.length();
                if len3 > 1e-30 && len3.is_finite() {
                    shadow3.set_pos(pt_m.pos + (sep3_orth / len3) * LY_EPS);
                } else if len3.is_finite() {
                    shadow3.set_pos(pt_m.pos + d1_hat.cross(d2_hat).normalize() * LY_EPS);
                }
            }
        }
    }

    // --- Test phase ----------------------------------------------------------
    let mut ly1_sum   = 0.0f64;
    let mut ly2_sum   = 0.0f64;
    let mut ly3_sum   = 0.0f64;
    let mut ly1_count = 0u32;
    let mut ly2_count = 0u32;
    let mut ly3_count = 0u32;
    let mut bb_min    = Vec3::splat(f32::MAX);
    let mut bb_max    = Vec3::splat(f32::MIN);
    // Speed coefficient-of-variation: limit cycles move at near-constant speed;
    // chaotic attractors have strongly varying speed.
    let mut spd_sum   = 0.0f64;
    // Unique-cell coverage: a period-P orbit visits exactly P distinct positions.
    // Chaotic attractors continuously visit new positions throughout the test.
    // We track the second half of the test (once the bounding box is stable).
    let mut uniq_cells: std::collections::HashSet<(i32, i32, i32)> = Default::default();
    let mut spd_sq    = 0.0f64;
    let mut spd_n     = 0u32;
    // Recurrence reference: a period-P orbit returns to this position every P steps.
    // Chaotic orbits never return within a small fraction of the bounding box.
    let mut recurrence_ref = Vec3::ZERO;
    fixed_streak = 0;
    prev = main.pos();

    for i in 0..TEST {
        if i % CHECK_INT == 0 && stop.load(Ordering::Relaxed) { return None; }

        let pt_m = match main.step() {
            None => return None,
            Some(pt) => pt,
        };
        if !pt_m.pos.is_finite() || pt_m.pos.length() > BOUND { return None; }

        let disp = (pt_m.pos - prev).length();
        if disp < 1e-7 {
            fixed_streak += 1;
            if fixed_streak > 5 { return None; }
        } else {
            fixed_streak = 0;
        }
        if i == 200 {
            recurrence_ref = pt_m.pos;
        }
        if i >= 200 {
            spd_sum += disp as f64;
            spd_sq  += (disp * disp) as f64;
            spd_n   += 1;
        }

        // Periodic-orbit recurrence check: a period-P orbit returns to the
        // reference position every P steps with near-zero distance.  Chaotic
        // orbits never return within 0.1 % of the bounding-box diagonal.
        if i >= 300 && i % 100 == 0 {
            let bb_diag = (bb_max - bb_min).length().max(0.2);
            if (pt_m.pos - recurrence_ref).length() < 0.001 * bb_diag {
                return None;
            }
        }
        prev = pt_m.pos;

        // --- shadow1 → λ₁ ---
        if let Some(pt_s1) = shadow1.step() {
            let sep1 = pt_s1.pos - pt_m.pos;
            let len1 = sep1.length();
            if len1 > 0.0 && len1.is_finite() {
                ly1_sum   += (len1 / LY_EPS).ln() as f64;
                ly1_count += 1;
                d1_hat     = sep1 / len1;
                shadow1.set_pos(pt_m.pos + d1_hat * LY_EPS);
            }
        }

        // --- shadow2 → λ₂ (Gram-Schmidt ⊥ shadow1) ---
        if let Some(pt_s2) = shadow2.step() {
            let sep2      = pt_s2.pos - pt_m.pos;
            let sep2_orth = sep2 - sep2.dot(d1_hat) * d1_hat;
            let len2      = sep2_orth.length();
            if len2 > 1e-30 && len2.is_finite() {
                ly2_sum   += (len2 / LY_EPS).ln() as f64;
                ly2_count += 1;
                d2_hat     = sep2_orth / len2;
                shadow2.set_pos(pt_m.pos + d2_hat * LY_EPS);
            } else if len2.is_finite() {
                let perp = if d1_hat.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                d2_hat = (perp - perp.dot(d1_hat) * d1_hat).normalize();
                shadow2.set_pos(pt_m.pos + d2_hat * LY_EPS);
            }
        }

        // --- shadow3 → λ₃ (Gram-Schmidt ⊥ shadow1 and shadow2) ---
        // Only needed for flow attractors; skip for maps to save computation.
        if is_flow {
            if let Some(pt_s3) = shadow3.step() {
                let sep3      = pt_s3.pos - pt_m.pos;
                let sep3_orth = sep3
                    - sep3.dot(d1_hat) * d1_hat
                    - sep3.dot(d2_hat) * d2_hat;
                let len3 = sep3_orth.length();
                if len3 > 1e-30 && len3.is_finite() {
                    ly3_sum   += (len3 / LY_EPS).ln() as f64;
                    ly3_count += 1;
                    let d3_hat = sep3_orth / len3;
                    shadow3.set_pos(pt_m.pos + d3_hat * LY_EPS);
                } else if len3.is_finite() {
                    let d3_new = d1_hat.cross(d2_hat).normalize();
                    shadow3.set_pos(pt_m.pos + d3_new * LY_EPS);
                }
            }
        }

        // Early exit (Sprott §5.6).
        // Maps:  exit if λ₁ is clearly negative (converging/periodic orbit).
        // Flows: exit if λ₁ is near zero after shadow alignment (limit cycle).
        if i == EARLY_CHK - 1 && ly1_count > 0 {
            let ly1_so_far = (ly1_sum / ly1_count as f64) as f32;
            let thresh = if is_flow { 0.001 } else { LY_EARLY_THRESH };
            if ly1_so_far < thresh { return None; }
        }

        if i >= 200 {
            bb_min = bb_min.min(pt_m.pos);
            bb_max = bb_max.max(pt_m.pos);
        }

        // Accumulate unique cells in the second half, once the bounding box
        // is stable enough to use as a reference for cell size.
        if i >= TEST / 2 {
            let cs = (bb_max - bb_min).max_element().max(0.001) / 50.0;
            uniq_cells.insert((
                ((pt_m.pos.x - bb_min.x) / cs).round() as i32,
                ((pt_m.pos.y - bb_min.y) / cs).round() as i32,
                ((pt_m.pos.z - bb_min.z) / cs).round() as i32,
            ));
        }
    }

    if ly1_count == 0 { return None; }
    let ly1 = (ly1_sum / ly1_count as f64) as f32;

    // --- Kaplan-Yorke dimension, with the correct formula per attractor type -
    let d_ky = if is_flow {
        // ODE flow: D_KY = 2 + λ₁/|λ₃|.
        // A true limit cycle has λ₁ = 0 → D_KY = 2.0, rejected by FLOW_DIM_MIN.
        if ly1 <= 0.0 { return None; }
        if ly3_count == 0 { return None; }
        let ly3 = (ly3_sum / ly3_count as f64) as f32;
        if ly3 >= 0.0 { return None; } // not dissipative
        2.0_f32 + ly1 / (-ly3)
    } else {
        // Discrete map: D_KY = 1 + λ₁/|λ₂|.
        if ly1 < LY_MIN || ly1 > LY_MAX { return None; }
        if ly2_count == 0 { return None; }
        let ly2 = (ly2_sum / ly2_count as f64) as f32;
        if ly2 >= 0.0 { return None; }
        1.0_f32 + ly1 / (-ly2)
    };

    let (dim_min, dim_max) = if is_flow {
        (FLOW_DIM_MIN, FLOW_DIM_MAX)
    } else {
        (DIM_MIN, DIM_MAX)
    };
    if d_ky < dim_min || d_ky > dim_max { return None; }

    // Require meaningful physical extent in at least 2 dimensions.
    let bb = bb_max - bb_min;
    let wide = [bb.x, bb.y, bb.z].iter().filter(|&&v| v > 0.1).count();
    if wide < 2 { return None; }

    // Reject short-period orbits: period-P orbits visit exactly P distinct
    // positions; genuine chaotic attractors visit far more than 200 in 4000 steps.
    if uniq_cells.len() < 200 { return None; }

    // Speed coefficient of variation: limit cycles move at near-constant speed
    // (CV ≈ 0); chaotic attractors have strongly varying speed (CV >> 0.05).
    // This catches circular/near-circular limit cycles that slip past the
    // Lyapunov and D_KY filters due to incomplete shadow alignment.
    if spd_n > 0 {
        let mean = spd_sum / spd_n as f64;
        let var  = (spd_sq / spd_n as f64 - mean * mean).max(0.0);
        let cv   = (var.sqrt() / mean.max(1e-20)) as f32;
        if cv < 0.05 { return None; }
    }

    Some((bb_min, bb_max))
}

// ---- Background Lyapunov / D_KY display worker -----------------------------

pub struct LyapunovWorker {
    pub result_rx: Receiver<(f32, f32)>,
    stop:          Arc<AtomicBool>,
    handle:        Option<JoinHandle<()>>,
}

impl LyapunovWorker {
    /// Spawn a single background thread that computes (λ₁, D_KY) for the
    /// current attractor config and sends the result once ready.
    pub fn spawn(config: AttractorConfig) -> Self {
        let (tx, result_rx) = bounded::<(f32, f32)>(1);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);

        let handle = thread::spawn(move || {
            if let Some(v) = metrics_dispatch(
                config.attractor_type, &config.params, &stop_flag,
            ) {
                let _ = tx.try_send(v);
            }
        });

        LyapunovWorker { result_rx, stop, handle: Some(handle) }
    }
}

impl Drop for LyapunovWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() { let _ = h.join(); }
    }
}

fn metrics_dispatch(
    t:      AttractorType,
    params: &[f32],
    stop:   &AtomicBool,
) -> Option<(f32, f32)> {
    let is_flow = t.is_flow();
    macro_rules! probe {
        ($T:ty) => {{
            let mut main = <$T>::new(); main.reset(params);
            let mut s1   = <$T>::new(); s1.reset(params);
            let mut s2   = <$T>::new(); s2.reset(params);
            let mut s3   = <$T>::new(); s3.reset(params);
            run_metrics(&mut main, &mut s1, &mut s2, &mut s3, is_flow, stop)
        }};
    }
    match t {
        AttractorType::Lorenz      => probe!(Lorenz),
        AttractorType::Lorenz84    => probe!(Lorenz84),
        AttractorType::Rossler     => probe!(Rossler),
        AttractorType::Thomas      => probe!(Thomas),
        AttractorType::ChaoticFlow => probe!(ChaoticFlow),
        AttractorType::Clifford    => probe!(Clifford),
        AttractorType::Icon        => probe!(Icon),
        AttractorType::IconB       => probe!(IconB),
        AttractorType::Pickover    => probe!(Pickover),
        AttractorType::PolyA       => probe!(PolyA),
        AttractorType::PolyAbs     => probe!(PolyAbs),
        AttractorType::PolyB       => probe!(PolyB),
        AttractorType::PolyC       => probe!(PolyC),
        AttractorType::PolyPow     => probe!(PolyPow),
        AttractorType::PolySin     => probe!(PolySin),
        AttractorType::PolySprott  => probe!(PolySprott),
    }
}

/// Compute (λ₁, D_KY) for display. Fewer steps than run_test for responsiveness;
/// no acceptance filters — returns raw values regardless of chaos quality.
fn run_metrics<A: Attractor>(
    main:    &mut A,
    shadow1: &mut A,
    shadow2: &mut A,
    shadow3: &mut A,
    is_flow: bool,
    stop:    &AtomicBool,
) -> Option<(f32, f32)> {
    const WARMUP:       usize = 200;
    const SHAD_WARMUP:  usize = 2_000;
    const TEST:         usize = 5_000;
    const LY_EPS:       f32   = 1e-4;
    const BOUND:        f32   = 1e6;
    const CHECK_INT:    usize = 500;

    let mut fixed_streak = 0usize;
    let mut prev = main.pos();
    for i in 0..WARMUP {
        if i % CHECK_INT == 0 && stop.load(Ordering::Relaxed) { return None; }
        match main.step() {
            None => return None,
            Some(pt) => {
                if !pt.pos.is_finite() || pt.pos.length() > BOUND { return None; }
                let d = (pt.pos - prev).length();
                if d < 1e-7 { fixed_streak += 1; if fixed_streak > 5 { return None; } }
                else { fixed_streak = 0; }
                prev = pt.pos;
            }
        }
    }

    shadow1.set_pos(main.pos() + Vec3::X * LY_EPS);
    shadow2.set_pos(main.pos() + Vec3::Y * LY_EPS);
    shadow3.set_pos(main.pos() + Vec3::Z * LY_EPS);
    let mut d1_hat = Vec3::X;
    let mut d2_hat = Vec3::Y;

    for _ in 0..SHAD_WARMUP {
        let pt_m = match main.step() { None => return None, Some(p) => p };
        if !pt_m.pos.is_finite() || pt_m.pos.length() > BOUND { return None; }
        if let Some(s) = shadow1.step() {
            let sep = s.pos - pt_m.pos; let len = sep.length();
            if len > 0.0 && len.is_finite() { d1_hat = sep/len; shadow1.set_pos(pt_m.pos + d1_hat*LY_EPS); }
        }
        if let Some(s) = shadow2.step() {
            let sep = s.pos - pt_m.pos;
            let orth = sep - sep.dot(d1_hat)*d1_hat; let len = orth.length();
            if len > 1e-30 && len.is_finite() { d2_hat = orth/len; shadow2.set_pos(pt_m.pos + d2_hat*LY_EPS); }
        }
        if is_flow {
            if let Some(s) = shadow3.step() {
                let sep = s.pos - pt_m.pos;
                let orth = sep - sep.dot(d1_hat)*d1_hat - sep.dot(d2_hat)*d2_hat;
                let len = orth.length();
                if len > 1e-30 && len.is_finite() { shadow3.set_pos(pt_m.pos + (orth/len)*LY_EPS); }
                else if len.is_finite() { shadow3.set_pos(pt_m.pos + d1_hat.cross(d2_hat).normalize()*LY_EPS); }
            }
        }
    }

    let mut ly1_sum = 0f64; let mut ly1_n = 0u32;
    let mut ly2_sum = 0f64; let mut ly2_n = 0u32;
    let mut ly3_sum = 0f64; let mut ly3_n = 0u32;
    fixed_streak = 0; prev = main.pos();

    for i in 0..TEST {
        if i % CHECK_INT == 0 && stop.load(Ordering::Relaxed) { return None; }
        let pt_m = match main.step() { None => return None, Some(p) => p };
        if !pt_m.pos.is_finite() || pt_m.pos.length() > BOUND { return None; }
        let d = (pt_m.pos - prev).length();
        if d < 1e-7 { fixed_streak += 1; if fixed_streak > 5 { return None; } }
        else { fixed_streak = 0; }
        prev = pt_m.pos;

        if let Some(s) = shadow1.step() {
            let sep = s.pos - pt_m.pos; let len = sep.length();
            if len > 0.0 && len.is_finite() {
                ly1_sum += (len/LY_EPS).ln() as f64; ly1_n += 1;
                d1_hat = sep/len; shadow1.set_pos(pt_m.pos + d1_hat*LY_EPS);
            }
        }
        if let Some(s) = shadow2.step() {
            let sep = s.pos - pt_m.pos;
            let orth = sep - sep.dot(d1_hat)*d1_hat; let len = orth.length();
            if len > 1e-30 && len.is_finite() {
                ly2_sum += (len/LY_EPS).ln() as f64; ly2_n += 1;
                d2_hat = orth/len; shadow2.set_pos(pt_m.pos + d2_hat*LY_EPS);
            } else if len.is_finite() {
                let perp = if d1_hat.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                d2_hat = (perp - perp.dot(d1_hat)*d1_hat).normalize();
                shadow2.set_pos(pt_m.pos + d2_hat*LY_EPS);
            }
        }
        if is_flow {
            if let Some(s) = shadow3.step() {
                let sep = s.pos - pt_m.pos;
                let orth = sep - sep.dot(d1_hat)*d1_hat - sep.dot(d2_hat)*d2_hat;
                let len = orth.length();
                if len > 1e-30 && len.is_finite() {
                    ly3_sum += (len/LY_EPS).ln() as f64; ly3_n += 1;
                    shadow3.set_pos(pt_m.pos + (orth/len)*LY_EPS);
                } else if len.is_finite() {
                    shadow3.set_pos(pt_m.pos + d1_hat.cross(d2_hat).normalize()*LY_EPS);
                }
            }
        }
    }

    if ly1_n == 0 { return None; }
    let ly1 = (ly1_sum / ly1_n as f64) as f32;

    let d_ky = if is_flow {
        if ly3_n == 0 { return None; }
        let ly3 = (ly3_sum / ly3_n as f64) as f32;
        if ly3 >= 0.0 { return None; }
        2.0_f32 + ly1 / (-ly3)
    } else {
        if ly2_n == 0 { return None; }
        let ly2 = (ly2_sum / ly2_n as f64) as f32;
        if ly2 >= 0.0 { return None; }
        1.0_f32 + ly1 / (-ly2)
    };

    Some((ly1, d_ky))
}
