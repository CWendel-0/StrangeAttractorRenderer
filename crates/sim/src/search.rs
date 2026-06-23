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
use crate::genesio_tesi::GenesioTesi;
use crate::arneodo::Arneodo;
use crate::chen_celikovsky::ChenCelikovsky;
use crate::shimizu_morioka::ShimizuMorioka;
use crate::three_cells_cnn::ThreeCellsCnn;
use crate::rucklidge::Rucklidge;
use crate::rayleigh_benard::RayleighBenard;
use crate::burke_shaw::BurkeShaw;
use crate::sakarya::Sakarya;
use crate::strizhak_kawczynski::StrizhakKawczynski;
use crate::bouali::Bouali;
use crate::halvorsen::Halvorsen;
use crate::aizawa::Aizawa;
use crate::dequan_li::DequanLi;
use crate::hadley::Hadley;
use crate::nose_hoover::NoseHoover;
use crate::newton_leipnik::NewtonLeipnik;
use crate::finance::Finance;
use crate::chen_lee::ChenLee;
use crate::sprott_linz_b::SprottLinzB;
use crate::sprott_linz_c::SprottLinzC;
use crate::sprott_linz_d::SprottLinzD;
use crate::sprott_linz_e::SprottLinzE;
use crate::sprott_linz_f::SprottLinzF;
use crate::sprott_linz_g::SprottLinzG;
use crate::sprott_linz_h::SprottLinzH;
use crate::sprott_linz_i::SprottLinzI;
use crate::sprott_linz_j::SprottLinzJ;
use crate::sprott_linz_k::SprottLinzK;
use crate::sprott_linz_l::SprottLinzL;
use crate::sprott_linz_m::SprottLinzM;
use crate::sprott_linz_n::SprottLinzN;
use crate::sprott_linz_o::SprottLinzO;
use crate::sprott_linz_p::SprottLinzP;
use crate::sprott_linz_q::SprottLinzQ;
use crate::sprott_linz_r::SprottLinzR;
use crate::sprott_linz_s::SprottLinzS;
use crate::act::Act;
use crate::anishchenko_astakhov::AnishchenkoAstakhov;
use crate::arnold::Arnold;
use crate::bouali_iii::BoualiIii;
use crate::burgers::Burgers;
use crate::business_cycle_map::BusinessCycleMap;
use crate::cathala::Cathala;
use crate::chua_cubic::ChuaCubic;
use crate::coullet::Coullet;
use crate::dadras::Dadras;
use crate::elhadj_sprott::ElhadjSprott;
use crate::elhadj_sprott_a::ElhadjSprottA;
use crate::elhadj_sprott_c_map::ElhadjSprottCMap;
use crate::four_wing::FourWing;
use crate::four_wing2::FourWing2;
use crate::four_wing3::FourWing3;
use crate::gingerbread::Gingerbread;
use crate::gumowski_mira::GumowskiMira;
use crate::hca::Hca;
use crate::heagy_hammel::HeagyHammel;
use crate::hopalong::Hopalong;
use crate::ikeda::Ikeda;
use crate::ikeda1::Ikeda1;
use crate::liu_chen::LiuChen;
use crate::lorenz_mod1::LorenzMod1;
use crate::lorenz_mod2::LorenzMod2;
use crate::lue_chen::LueChen;
use crate::mira::Mira;
use crate::modified_lozi::ModifiedLozi;
use crate::multi_chua_ii::MultiChuaII;
use crate::multifold_henon::MultifoldHenon;
use crate::popcorn::Popcorn;
use crate::popcorn2::Popcorn2;
use crate::qi_3d::Qi3D;
use crate::qi_chen::QiChen;
use crate::robinson::Robinson;
use crate::serpentine::Serpentine;
use crate::strelkova_anishchenko::StrelkovaAnishchenko;
use crate::tsucs_1::Tsucs1;
use crate::tsucs_2::Tsucs2;
use crate::wang_sun::WangSun;
use crate::de_jong::DeJong;
use crate::mac_millan::MacMillan;
use crate::marotto_lorenz::MarottoLorenz;
use crate::maynard_smith::MaynardSmith;
use crate::nishikawa_kaneko::NishikawaKaneko;
use crate::q_henon2::QHenon2;
use crate::rulkov_map::RulkovMap;
use crate::sine_delay::SineDelay;
use crate::sine_sine_map::SineSineMap;
use crate::svensson::Svensson;
use crate::tinkerbell::Tinkerbell;
use crate::yang_cao::YangCao;
use crate::zhou::Zhou;
use crate::zhou_chen::ZhouChen;

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
        AttractorType::GenesioTesi => probe!(GenesioTesi),
        AttractorType::Arneodo     => probe!(Arneodo),
        AttractorType::ChenCelikovsky     => probe!(ChenCelikovsky),
        AttractorType::ShimizuMorioka     => probe!(ShimizuMorioka),
        AttractorType::ThreeCellsCnn      => probe!(ThreeCellsCnn),
        AttractorType::Rucklidge          => probe!(Rucklidge),
        AttractorType::RayleighBenard     => probe!(RayleighBenard),
        AttractorType::BurkeShaw          => probe!(BurkeShaw),
        AttractorType::Sakarya            => probe!(Sakarya),
        AttractorType::StrizhakKawczynski => probe!(StrizhakKawczynski),
        AttractorType::Bouali             => probe!(Bouali),
        AttractorType::Halvorsen          => probe!(Halvorsen),
        AttractorType::Aizawa             => probe!(Aizawa),
        AttractorType::DequanLi           => probe!(DequanLi),
        AttractorType::Hadley             => probe!(Hadley),
        AttractorType::NoseHoover         => probe!(NoseHoover),
        AttractorType::NewtonLeipnik      => probe!(NewtonLeipnik),
        AttractorType::Finance            => probe!(Finance),
        AttractorType::ChenLee            => probe!(ChenLee),
        AttractorType::SprottLinzB => probe!(SprottLinzB),
        AttractorType::SprottLinzC => probe!(SprottLinzC),
        AttractorType::SprottLinzD => probe!(SprottLinzD),
        AttractorType::SprottLinzE => probe!(SprottLinzE),
        AttractorType::SprottLinzF => probe!(SprottLinzF),
        AttractorType::SprottLinzG => probe!(SprottLinzG),
        AttractorType::SprottLinzH => probe!(SprottLinzH),
        AttractorType::SprottLinzI => probe!(SprottLinzI),
        AttractorType::SprottLinzJ => probe!(SprottLinzJ),
        AttractorType::SprottLinzK => probe!(SprottLinzK),
        AttractorType::SprottLinzL => probe!(SprottLinzL),
        AttractorType::SprottLinzM => probe!(SprottLinzM),
        AttractorType::SprottLinzN => probe!(SprottLinzN),
        AttractorType::SprottLinzO => probe!(SprottLinzO),
        AttractorType::SprottLinzP => probe!(SprottLinzP),
        AttractorType::SprottLinzQ => probe!(SprottLinzQ),
        AttractorType::SprottLinzR => probe!(SprottLinzR),
        AttractorType::SprottLinzS => probe!(SprottLinzS),
        AttractorType::Act => probe!(Act),
        AttractorType::AnishchenkoAstakhov => probe!(AnishchenkoAstakhov),
        AttractorType::Arnold => probe!(Arnold),
        AttractorType::BoualiIii => probe!(BoualiIii),
        AttractorType::Burgers => probe!(Burgers),
        AttractorType::BusinessCycleMap => probe!(BusinessCycleMap),
        AttractorType::Cathala => probe!(Cathala),
        AttractorType::ChuaCubic => probe!(ChuaCubic),
        AttractorType::Coullet => probe!(Coullet),
        AttractorType::Dadras => probe!(Dadras),
        AttractorType::ElhadjSprott => probe!(ElhadjSprott),
        AttractorType::ElhadjSprottA => probe!(ElhadjSprottA),
        AttractorType::ElhadjSprottCMap => probe!(ElhadjSprottCMap),
        AttractorType::FourWing => probe!(FourWing),
        AttractorType::FourWing2 => probe!(FourWing2),
        AttractorType::FourWing3 => probe!(FourWing3),
        AttractorType::Gingerbread => probe!(Gingerbread),
        AttractorType::GumowskiMira => probe!(GumowskiMira),
        AttractorType::Hca => probe!(Hca),
        AttractorType::HeagyHammel => probe!(HeagyHammel),
        AttractorType::Hopalong => probe!(Hopalong),
        AttractorType::Ikeda => probe!(Ikeda),
        AttractorType::Ikeda1 => probe!(Ikeda1),
        AttractorType::LiuChen => probe!(LiuChen),
        AttractorType::LorenzMod1 => probe!(LorenzMod1),
        AttractorType::LorenzMod2 => probe!(LorenzMod2),
        AttractorType::LueChen => probe!(LueChen),
        AttractorType::Mira => probe!(Mira),
        AttractorType::ModifiedLozi => probe!(ModifiedLozi),
        AttractorType::MultiChuaII => probe!(MultiChuaII),
        AttractorType::MultifoldHenon => probe!(MultifoldHenon),
        AttractorType::Popcorn => probe!(Popcorn),
        AttractorType::Popcorn2 => probe!(Popcorn2),
        AttractorType::Qi3D => probe!(Qi3D),
        AttractorType::QiChen => probe!(QiChen),
        AttractorType::Robinson => probe!(Robinson),
        AttractorType::Serpentine => probe!(Serpentine),
        AttractorType::StrelkovaAnishchenko => probe!(StrelkovaAnishchenko),
        AttractorType::Tsucs1 => probe!(Tsucs1),
        AttractorType::Tsucs2 => probe!(Tsucs2),
        AttractorType::WangSun => probe!(WangSun),
        AttractorType::DeJong => probe!(DeJong),
        AttractorType::MacMillan => probe!(MacMillan),
        AttractorType::MarottoLorenz => probe!(MarottoLorenz),
        AttractorType::MaynardSmith => probe!(MaynardSmith),
        AttractorType::NishikawaKaneko => probe!(NishikawaKaneko),
        AttractorType::QHenon2 => probe!(QHenon2),
        AttractorType::RulkovMap => probe!(RulkovMap),
        AttractorType::SineDelay => probe!(SineDelay),
        AttractorType::SineSineMap => probe!(SineSineMap),
        AttractorType::Svensson => probe!(Svensson),
        AttractorType::Tinkerbell => probe!(Tinkerbell),
        AttractorType::YangCao => probe!(YangCao),
        AttractorType::Zhou => probe!(Zhou),
        AttractorType::ZhouChen => probe!(ZhouChen),
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
    // Map thresholds.  LY_MAX was originally 0.3, calibrated against a handful of
    // weakly-chaotic Hénon/Clifford-style maps; the much wider variety of map
    // families ported since then includes several legitimately good-looking
    // attractors (e.g. Arnold's cat map, Elhadj-Sprott A) whose true λ₁ is far
    // higher (measured 0.3–8.5+) without looking like unstructured noise — the
    // dimension/recurrence/speed-CV/uniqueness filters below still gate quality.
    const LY_MIN:          f32   = 0.03;
    const LY_MAX:          f32   = 10.0;
    // DIM_MIN was 1.5; quasi-periodically-forced maps (a "+const mod 1" phase
    // coordinate riding alongside the chaotic one, e.g. Q-Henon 2) measure a
    // legitimately lower Kaplan-Yorke dimension since part of the state is a
    // pure rotation, not chaotic — 1.05 still excludes near-1D limit cycles.
    const DIM_MIN:         f32   = 1.05;
    const DIM_MAX:         f32   = 10.0;
    // Flow thresholds (D_KY = 2 + λ₁/|λ₃|; limit cycles give exactly 2.0).
    // FLOW_DIM_MIN was 2.05; a few legitimately-chaotic-but-weak flows (e.g.
    // ACT) sit only marginally above 2.0 (measured ~2.02-2.04) — still a
    // meaningful margin over an exact limit cycle, just not a strongly mixing
    // one.
    const FLOW_DIM_MIN:    f32   = 2.02;
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
        // Lyapunov bookkeeping always operates in the attractor's true phase-space
        // coordinates (`pos()`), not the rendered point (`step()`'s `Point.pos`).
        // These coincide for almost every attractor, but a few (e.g. Yang-Cao)
        // render a derived projection of the internal state; using the rendered
        // point here would feed mismatched-scale coordinates into `set_pos()`.
        let main_pos = main.pos();

        if shadow1.step().is_some() {
            let sep1 = shadow1.pos() - main_pos;
            let len1 = sep1.length();
            if len1 > 0.0 && len1.is_finite() {
                d1_hat = sep1 / len1;
                shadow1.set_pos(main_pos + d1_hat * LY_EPS);
            } else {
                // Collapsed onto main (e.g. landed on a critical point of the
                // map where separation underflows below f32 precision) —
                // re-seed so it doesn't stay locked to main forever.
                shadow1.set_pos(main_pos + d1_hat * LY_EPS);
            }
        } else {
            // Shadow escaped (e.g. a near-boundary excursion in a map whose
            // formula is only well-behaved inside a bounded domain) and is
            // permanently stuck returning None from its last finite state.
            // Re-seed it next to main along the last known direction instead
            // of leaving it dead for the rest of the run.
            shadow1.set_pos(main_pos + d1_hat * LY_EPS);
        }
        if shadow2.step().is_some() {
            let sep2      = shadow2.pos() - main_pos;
            let sep2_orth = sep2 - sep2.dot(d1_hat) * d1_hat;
            let len2      = sep2_orth.length();
            if len2 > 1e-30 && len2.is_finite() {
                d2_hat = sep2_orth / len2;
                shadow2.set_pos(main_pos + d2_hat * LY_EPS);
            } else if len2.is_finite() {
                let perp = if d1_hat.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                d2_hat = (perp - perp.dot(d1_hat) * d1_hat).normalize();
                shadow2.set_pos(main_pos + d2_hat * LY_EPS);
            }
        } else {
            shadow2.set_pos(main_pos + d2_hat * LY_EPS);
        }
        if is_flow {
            if shadow3.step().is_some() {
                let sep3      = shadow3.pos() - main_pos;
                let sep3_orth = sep3
                    - sep3.dot(d1_hat) * d1_hat
                    - sep3.dot(d2_hat) * d2_hat;
                let len3 = sep3_orth.length();
                if len3 > 1e-30 && len3.is_finite() {
                    shadow3.set_pos(main_pos + (sep3_orth / len3) * LY_EPS);
                } else if len3.is_finite() {
                    shadow3.set_pos(main_pos + d1_hat.cross(d2_hat).normalize() * LY_EPS);
                }
            } else {
                shadow3.set_pos(main_pos + d1_hat.cross(d2_hat).normalize() * LY_EPS);
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
    let mut recurrence_streak = 0u32;
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
        // Require two consecutive 100-step checkpoints to hit before rejecting:
        // maps with a deliberately periodic phase coordinate (e.g. a "+const
        // mod 1" rotation driving otherwise-chaotic dynamics, as in Heagy-Hammel
        // or Nishikawa-Kaneko) can land close to the reference once by sheer
        // coincidence in the chaotic coordinate while the rotating coordinate
        // cycles back — a true period-P orbit hits on every checkpoint, not
        // just one.
        if i >= 300 && i % 100 == 0 {
            let bb_diag = (bb_max - bb_min).length().max(0.2);
            if (pt_m.pos - recurrence_ref).length() < 0.001 * bb_diag {
                recurrence_streak += 1;
                if recurrence_streak >= 2 {
                    return None;
                }
            } else {
                recurrence_streak = 0;
            }
        }
        prev = pt_m.pos;
        // See the comment in the shadow-warmup loop above: Lyapunov bookkeeping
        // uses `pos()` (true phase-space state), not the rendered `step()` point.
        let main_pos = main.pos();

        // --- shadow1 → λ₁ ---
        if shadow1.step().is_some() {
            let sep1 = shadow1.pos() - main_pos;
            let len1 = sep1.length();
            if len1 > 0.0 && len1.is_finite() {
                ly1_sum   += (len1 / LY_EPS).ln() as f64;
                ly1_count += 1;
                d1_hat     = sep1 / len1;
                shadow1.set_pos(main_pos + d1_hat * LY_EPS);
            } else {
                // Separation underflowed to exactly 0 (or went non-finite)
                // without the shadow's own step failing — typically because
                // the orbit passed through a critical point of the map (e.g.
                // a 1D logistic-style fold, where derivative ≈ 0 collapses
                // an f32-scale separation below precision in one step).
                // Once collapsed, a deterministic map keeps shadow1 locked
                // exactly onto main forever; re-seed it next to main so it
                // can resume tracking instead of permanently zeroing ly1.
                shadow1.set_pos(main_pos + d1_hat * LY_EPS);
            }
        } else {
            // See the warmup loop: re-seed an escaped/stuck shadow rather
            // than letting it stay dead (and ly1_count stuck at 0) forever.
            shadow1.set_pos(main_pos + d1_hat * LY_EPS);
        }

        // --- shadow2 → λ₂ (Gram-Schmidt ⊥ shadow1) ---
        if shadow2.step().is_some() {
            let sep2      = shadow2.pos() - main_pos;
            let sep2_orth = sep2 - sep2.dot(d1_hat) * d1_hat;
            let len2      = sep2_orth.length();
            if len2 > 1e-30 && len2.is_finite() {
                ly2_sum   += (len2 / LY_EPS).ln() as f64;
                ly2_count += 1;
                d2_hat     = sep2_orth / len2;
                shadow2.set_pos(main_pos + d2_hat * LY_EPS);
            } else if len2.is_finite() {
                let perp = if d1_hat.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                d2_hat = (perp - perp.dot(d1_hat) * d1_hat).normalize();
                shadow2.set_pos(main_pos + d2_hat * LY_EPS);
            }
        } else {
            shadow2.set_pos(main_pos + d2_hat * LY_EPS);
        }

        // --- shadow3 → λ₃ (Gram-Schmidt ⊥ shadow1 and shadow2) ---
        // Only needed for flow attractors; skip for maps to save computation.
        if is_flow {
            if shadow3.step().is_some() {
                let sep3      = shadow3.pos() - main_pos;
                let sep3_orth = sep3
                    - sep3.dot(d1_hat) * d1_hat
                    - sep3.dot(d2_hat) * d2_hat;
                let len3 = sep3_orth.length();
                if len3 > 1e-30 && len3.is_finite() {
                    ly3_sum   += (len3 / LY_EPS).ln() as f64;
                    ly3_count += 1;
                    let d3_hat = sep3_orth / len3;
                    shadow3.set_pos(main_pos + d3_hat * LY_EPS);
                } else if len3.is_finite() {
                    let d3_new = d1_hat.cross(d2_hat).normalize();
                    shadow3.set_pos(main_pos + d3_new * LY_EPS);
                }
            } else {
                shadow3.set_pos(main_pos + d1_hat.cross(d2_hat).normalize() * LY_EPS);
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
        // Discrete map: D_KY = 1 + λ₁/|λ₂|, valid when λ₂ < 0 (a genuinely
        // dissipative/contracting direction, giving a thin fractal curve).
        // Some legitimately good-looking maps (e.g. Elhadj-Sprott A, Popcorn,
        // Serpentine) expand in BOTH directions on average — chaotic but
        // "area-filling" rather than a thin fractal — where λ₂ ≥ 0 makes the
        // Kaplan-Yorke formula inapplicable. Rather than reject outright,
        // treat that case as dimension-saturated (D_KY = DIM_MAX) and let the
        // other quality gates (recurrence/uniqueness/speed-CV/extent below)
        // decide whether it's actually a good attractor.
        if ly1 < LY_MIN || ly1 > LY_MAX { return None; }
        if ly2_count == 0 { return None; }
        let ly2 = (ly2_sum / ly2_count as f64) as f32;
        if ly2 >= 0.0 { DIM_MAX } else { 1.0_f32 + ly1 / (-ly2) }
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
        AttractorType::GenesioTesi => probe!(GenesioTesi),
        AttractorType::Arneodo     => probe!(Arneodo),
        AttractorType::ChenCelikovsky     => probe!(ChenCelikovsky),
        AttractorType::ShimizuMorioka     => probe!(ShimizuMorioka),
        AttractorType::ThreeCellsCnn      => probe!(ThreeCellsCnn),
        AttractorType::Rucklidge          => probe!(Rucklidge),
        AttractorType::RayleighBenard     => probe!(RayleighBenard),
        AttractorType::BurkeShaw          => probe!(BurkeShaw),
        AttractorType::Sakarya            => probe!(Sakarya),
        AttractorType::StrizhakKawczynski => probe!(StrizhakKawczynski),
        AttractorType::Bouali             => probe!(Bouali),
        AttractorType::Halvorsen          => probe!(Halvorsen),
        AttractorType::Aizawa             => probe!(Aizawa),
        AttractorType::DequanLi           => probe!(DequanLi),
        AttractorType::Hadley             => probe!(Hadley),
        AttractorType::NoseHoover         => probe!(NoseHoover),
        AttractorType::NewtonLeipnik      => probe!(NewtonLeipnik),
        AttractorType::Finance            => probe!(Finance),
        AttractorType::ChenLee            => probe!(ChenLee),
        AttractorType::SprottLinzB => probe!(SprottLinzB),
        AttractorType::SprottLinzC => probe!(SprottLinzC),
        AttractorType::SprottLinzD => probe!(SprottLinzD),
        AttractorType::SprottLinzE => probe!(SprottLinzE),
        AttractorType::SprottLinzF => probe!(SprottLinzF),
        AttractorType::SprottLinzG => probe!(SprottLinzG),
        AttractorType::SprottLinzH => probe!(SprottLinzH),
        AttractorType::SprottLinzI => probe!(SprottLinzI),
        AttractorType::SprottLinzJ => probe!(SprottLinzJ),
        AttractorType::SprottLinzK => probe!(SprottLinzK),
        AttractorType::SprottLinzL => probe!(SprottLinzL),
        AttractorType::SprottLinzM => probe!(SprottLinzM),
        AttractorType::SprottLinzN => probe!(SprottLinzN),
        AttractorType::SprottLinzO => probe!(SprottLinzO),
        AttractorType::SprottLinzP => probe!(SprottLinzP),
        AttractorType::SprottLinzQ => probe!(SprottLinzQ),
        AttractorType::SprottLinzR => probe!(SprottLinzR),
        AttractorType::SprottLinzS => probe!(SprottLinzS),
        AttractorType::Act => probe!(Act),
        AttractorType::AnishchenkoAstakhov => probe!(AnishchenkoAstakhov),
        AttractorType::Arnold => probe!(Arnold),
        AttractorType::BoualiIii => probe!(BoualiIii),
        AttractorType::Burgers => probe!(Burgers),
        AttractorType::BusinessCycleMap => probe!(BusinessCycleMap),
        AttractorType::Cathala => probe!(Cathala),
        AttractorType::ChuaCubic => probe!(ChuaCubic),
        AttractorType::Coullet => probe!(Coullet),
        AttractorType::Dadras => probe!(Dadras),
        AttractorType::ElhadjSprott => probe!(ElhadjSprott),
        AttractorType::ElhadjSprottA => probe!(ElhadjSprottA),
        AttractorType::ElhadjSprottCMap => probe!(ElhadjSprottCMap),
        AttractorType::FourWing => probe!(FourWing),
        AttractorType::FourWing2 => probe!(FourWing2),
        AttractorType::FourWing3 => probe!(FourWing3),
        AttractorType::Gingerbread => probe!(Gingerbread),
        AttractorType::GumowskiMira => probe!(GumowskiMira),
        AttractorType::Hca => probe!(Hca),
        AttractorType::HeagyHammel => probe!(HeagyHammel),
        AttractorType::Hopalong => probe!(Hopalong),
        AttractorType::Ikeda => probe!(Ikeda),
        AttractorType::Ikeda1 => probe!(Ikeda1),
        AttractorType::LiuChen => probe!(LiuChen),
        AttractorType::LorenzMod1 => probe!(LorenzMod1),
        AttractorType::LorenzMod2 => probe!(LorenzMod2),
        AttractorType::LueChen => probe!(LueChen),
        AttractorType::Mira => probe!(Mira),
        AttractorType::ModifiedLozi => probe!(ModifiedLozi),
        AttractorType::MultiChuaII => probe!(MultiChuaII),
        AttractorType::MultifoldHenon => probe!(MultifoldHenon),
        AttractorType::Popcorn => probe!(Popcorn),
        AttractorType::Popcorn2 => probe!(Popcorn2),
        AttractorType::Qi3D => probe!(Qi3D),
        AttractorType::QiChen => probe!(QiChen),
        AttractorType::Robinson => probe!(Robinson),
        AttractorType::Serpentine => probe!(Serpentine),
        AttractorType::StrelkovaAnishchenko => probe!(StrelkovaAnishchenko),
        AttractorType::Tsucs1 => probe!(Tsucs1),
        AttractorType::Tsucs2 => probe!(Tsucs2),
        AttractorType::WangSun => probe!(WangSun),
        AttractorType::DeJong => probe!(DeJong),
        AttractorType::MacMillan => probe!(MacMillan),
        AttractorType::MarottoLorenz => probe!(MarottoLorenz),
        AttractorType::MaynardSmith => probe!(MaynardSmith),
        AttractorType::NishikawaKaneko => probe!(NishikawaKaneko),
        AttractorType::QHenon2 => probe!(QHenon2),
        AttractorType::RulkovMap => probe!(RulkovMap),
        AttractorType::SineDelay => probe!(SineDelay),
        AttractorType::SineSineMap => probe!(SineSineMap),
        AttractorType::Svensson => probe!(Svensson),
        AttractorType::Tinkerbell => probe!(Tinkerbell),
        AttractorType::YangCao => probe!(YangCao),
        AttractorType::Zhou => probe!(Zhou),
        AttractorType::ZhouChen => probe!(ZhouChen),
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
        // See run_test: Lyapunov bookkeeping uses `pos()`, not the rendered point.
        let main_pos = main.pos();
        if shadow1.step().is_some() {
            let sep = shadow1.pos() - main_pos; let len = sep.length();
            if len > 0.0 && len.is_finite() { d1_hat = sep/len; shadow1.set_pos(main_pos + d1_hat*LY_EPS); }
            else { shadow1.set_pos(main_pos + d1_hat*LY_EPS); }
        } else {
            shadow1.set_pos(main_pos + d1_hat*LY_EPS);
        }
        if shadow2.step().is_some() {
            let sep = shadow2.pos() - main_pos;
            let orth = sep - sep.dot(d1_hat)*d1_hat; let len = orth.length();
            if len > 1e-30 && len.is_finite() { d2_hat = orth/len; shadow2.set_pos(main_pos + d2_hat*LY_EPS); }
        } else {
            shadow2.set_pos(main_pos + d2_hat*LY_EPS);
        }
        if is_flow {
            if shadow3.step().is_some() {
                let sep = shadow3.pos() - main_pos;
                let orth = sep - sep.dot(d1_hat)*d1_hat - sep.dot(d2_hat)*d2_hat;
                let len = orth.length();
                if len > 1e-30 && len.is_finite() { shadow3.set_pos(main_pos + (orth/len)*LY_EPS); }
                else if len.is_finite() { shadow3.set_pos(main_pos + d1_hat.cross(d2_hat).normalize()*LY_EPS); }
            } else {
                shadow3.set_pos(main_pos + d1_hat.cross(d2_hat).normalize()*LY_EPS);
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
        let main_pos = main.pos();

        if shadow1.step().is_some() {
            let sep = shadow1.pos() - main_pos; let len = sep.length();
            if len > 0.0 && len.is_finite() {
                ly1_sum += (len/LY_EPS).ln() as f64; ly1_n += 1;
                d1_hat = sep/len; shadow1.set_pos(main_pos + d1_hat*LY_EPS);
            } else {
                shadow1.set_pos(main_pos + d1_hat*LY_EPS);
            }
        } else {
            shadow1.set_pos(main_pos + d1_hat*LY_EPS);
        }
        if shadow2.step().is_some() {
            let sep = shadow2.pos() - main_pos;
            let orth = sep - sep.dot(d1_hat)*d1_hat; let len = orth.length();
            if len > 1e-30 && len.is_finite() {
                ly2_sum += (len/LY_EPS).ln() as f64; ly2_n += 1;
                d2_hat = orth/len; shadow2.set_pos(main_pos + d2_hat*LY_EPS);
            } else if len.is_finite() {
                let perp = if d1_hat.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                d2_hat = (perp - perp.dot(d1_hat)*d1_hat).normalize();
                shadow2.set_pos(main_pos + d2_hat*LY_EPS);
            }
        } else {
            shadow2.set_pos(main_pos + d2_hat*LY_EPS);
        }
        if is_flow {
            if shadow3.step().is_some() {
                let sep = shadow3.pos() - main_pos;
                let orth = sep - sep.dot(d1_hat)*d1_hat - sep.dot(d2_hat)*d2_hat;
                let len = orth.length();
                if len > 1e-30 && len.is_finite() {
                    ly3_sum += (len/LY_EPS).ln() as f64; ly3_n += 1;
                    shadow3.set_pos(main_pos + (orth/len)*LY_EPS);
                } else if len.is_finite() {
                    shadow3.set_pos(main_pos + d1_hat.cross(d2_hat).normalize()*LY_EPS);
                }
            } else {
                shadow3.set_pos(main_pos + d1_hat.cross(d2_hat).normalize()*LY_EPS);
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


#[cfg(test)]
mod tests {
    use super::*;

    /// Default parameters for these attractor types should pass the chaoticity
    /// filter in `test_params` — i.e. clicking "Randomize" (or just switching to
    /// the type) should reliably be able to find/recognize a good-looking result.
    /// Several of these were previously rejected outright by filter constants
    /// calibrated against a narrower, earlier set of attractor families; see the
    /// comments on LY_MAX/DIM_MIN/FLOW_DIM_MIN and the λ₂≥0 "area-filling chaos"
    /// branch above. Heagy-Hammel and Nishikawa-Kaneko are deliberately excluded:
    /// their shipped defaults sit right at the edge of chaos (λ₁ ≈ 0) and remain
    /// flaky even after the shadow-tracking fixes here.
    #[test]
    fn default_params_pass_chaoticity_filter() {
        const SHOULD_PASS: &[AttractorType] = &[
            AttractorType::Arnold,
            AttractorType::ElhadjSprottA,
            AttractorType::Popcorn,
            AttractorType::Serpentine,
            AttractorType::QHenon2,
            AttractorType::YangCao,
            AttractorType::ThreeCellsCnn,
            AttractorType::SprottLinzL,
            AttractorType::Act,
        ];
        let stop = AtomicBool::new(false);
        for &t in SHOULD_PASS {
            let descs = t.descriptors();
            let defaults: Vec<f32> = descs.iter().map(|d| d.default).collect();
            assert!(
                test_params(t, &defaults, &stop).is_some(),
                "{:?}: default params should pass the chaoticity filter", t
            );
        }
    }
}
