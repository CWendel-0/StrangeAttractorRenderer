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
                let atype     = config.attractor_type;
                let descs     = atype.descriptors();

                thread::spawn(move || {
                    let mut rng = Lcg64::seeded();

                    loop {
                        if stop_flag.load(Ordering::Relaxed) { break; }

                        let params = sample_params(descs, &mut rng);

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

fn sample_params(descs: &[crate::attractor::ParamDesc], rng: &mut Lcg64) -> Vec<f32> {
    use crate::attractor::ParamKind;
    descs.iter().map(|desc| {
        match &desc.kind {
            ParamKind::Continuous    => rng.range_f32(desc.min, desc.max),
            ParamKind::Integer       => rng.range_int(desc.min, desc.max),
            ParamKind::Enum(choices) => rng.range_int(0.0, choices.len() as f32 - 1.0),
        }
    }).collect()
}

// ---- Chaoticity test -------------------------------------------------------

fn test_params(t: AttractorType, params: &[f32], stop: &AtomicBool) -> Option<(Vec3, Vec3)> {
    macro_rules! probe {
        ($T:ty) => {{
            let mut main    = <$T>::new(); main.reset(params);
            let mut shadow1 = <$T>::new(); shadow1.reset(params);
            let mut shadow2 = <$T>::new(); shadow2.reset(params);
            run_test(&mut main, &mut shadow1, &mut shadow2, stop)
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

/// Sprott §3.3 / §3.4 shadow-orbit Lyapunov + Kaplan-Yorke dimension search.
///
/// Three trajectories run in parallel: `main`, `shadow1`, `shadow2`.
///
/// shadow1 gives λ₁ (largest Lyapunov exponent) via the standard shadow-orbit
/// renormalization (Sprott §3.3).
///
/// shadow2, kept orthogonal to shadow1 via Gram-Schmidt after each step, gives
/// λ₂ (the second Lyapunov exponent).  Together they yield the Kaplan-Yorke
/// fractal dimension (Sprott §3.4 / §8.3):
///
///   D_KY = 1 + λ₁ / |λ₂|   when λ₂ < 0  (D ∈ (1, 2) — fractal curve)
///   D_KY ≥ 2               when λ₂ ≥ 0  (volumetric attractor, rejected)
///
/// Aesthetic filter (Sprott §8.3, Figure 8-4):
///   Most visually interesting attractors have D_KY ∈ [DIM_MIN, DIM_MAX]
///   and λ₁ ∈ [LY_MIN, LY_MAX].
///
/// Early-exit (Sprott §5.6):
///   After EARLY_CHK Lyapunov steps, if λ₁ is clearly negative the orbit is
///   periodic or converging; skip the remaining test steps.
fn run_test<A: Attractor>(
    main:    &mut A,
    shadow1: &mut A,
    shadow2: &mut A,
    stop:    &AtomicBool,
) -> Option<(Vec3, Vec3)> {
    const WARMUP:          usize = 200;
    const TEST:            usize = 8_000;
    const EARLY_CHK:       usize = 500;
    const LY_EARLY_THRESH: f32   = -0.1;  // clearly non-chaotic → skip remaining steps
    const LY_EPS:          f32   = 1e-6;
    const LY_MIN:          f32   = 0.005; // Sprott §2.4 threshold
    const LY_MAX:          f32   = 3.0;
    const DIM_MIN:         f32   = 1.05;  // Sprott §8.3: aesthetically interesting range
    const DIM_MAX:         f32   = 1.9;
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

    // --- Place shadows at main + ε along orthogonal axes --------------------
    shadow1.set_pos(main.pos() + Vec3::X * LY_EPS);
    shadow2.set_pos(main.pos() + Vec3::Y * LY_EPS);

    // --- Test phase: λ₁, λ₂ via two-shadow Gram-Schmidt, + bounding box ----
    let mut ly1_sum   = 0.0f64;
    let mut ly2_sum   = 0.0f64;
    let mut ly1_count = 0u32;
    let mut ly2_count = 0u32;
    let mut bb_min    = Vec3::splat(f32::MAX);
    let mut bb_max    = Vec3::splat(f32::MIN);
    fixed_streak = 0;
    prev = main.pos();

    // Track the current renormalized direction of shadow1 so shadow2's
    // Gram-Schmidt can remove the shadow1 component each step.
    let mut d1_hat = Vec3::X;

    for i in 0..TEST {
        if i % CHECK_INT == 0 && stop.load(Ordering::Relaxed) { return None; }

        // Advance main
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
        prev = pt_m.pos;

        // --- shadow1: gives λ₁ ---
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

        // --- shadow2: Gram-Schmidt ⊥ shadow1, gives λ₂ ---
        if let Some(pt_s2) = shadow2.step() {
            let sep2      = pt_s2.pos - pt_m.pos;
            // Remove the shadow1 component; what remains is the expansion rate
            // in the direction orthogonal to the most-expanding direction → λ₂.
            let sep2_orth = sep2 - sep2.dot(d1_hat) * d1_hat;
            let len2      = sep2_orth.length();
            if len2 > 1e-30 && len2.is_finite() {
                ly2_sum   += (len2 / LY_EPS).ln() as f64;
                ly2_count += 1;
                let d2_hat = sep2_orth / len2;
                shadow2.set_pos(pt_m.pos + d2_hat * LY_EPS);
            } else if len2.is_finite() {
                // Shadow2 collapsed onto shadow1: reinitialize perpendicular to d1_hat.
                let perp = if d1_hat.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                let d2_new = (perp - perp.dot(d1_hat) * d1_hat).normalize();
                shadow2.set_pos(pt_m.pos + d2_new * LY_EPS);
            }
        }

        // Early exit: after EARLY_CHK steps a clearly negative λ₁ running
        // average means the orbit is periodic or converging (Sprott §5.6).
        if i == EARLY_CHK - 1 && ly1_count > 0 {
            let ly1_so_far = (ly1_sum / ly1_count as f64) as f32;
            if ly1_so_far < LY_EARLY_THRESH { return None; }
        }

        if i >= 200 {
            bb_min = bb_min.min(pt_m.pos);
            bb_max = bb_max.max(pt_m.pos);
        }
    }

    if ly1_count == 0 { return None; }

    let ly1 = (ly1_sum / ly1_count as f64) as f32;
    if ly1 < LY_MIN || ly1 > LY_MAX { return None; }

    // Kaplan-Yorke fractal dimension (Sprott §3.4 / §8.3).
    // Valid for D < 2: requires λ₂ < 0.  Volumetric attractors (λ₂ ≥ 0)
    // are less visually interesting and are rejected.
    let d_ky = if ly2_count > 0 {
        let ly2 = (ly2_sum / ly2_count as f64) as f32;
        if ly2 >= 0.0 {
            return None; // volumetric attractor: D ≥ 2
        }
        1.0_f32 + ly1 / (-ly2)
    } else {
        return None;
    };

    if d_ky < DIM_MIN || d_ky > DIM_MAX { return None; }

    // Require meaningful physical extent in at least 2 dimensions.
    let bb = bb_max - bb_min;
    let wide = [bb.x, bb.y, bb.z].iter().filter(|&&v| v > 0.1).count();
    if wide < 2 { return None; }

    Some((bb_min, bb_max))
}
