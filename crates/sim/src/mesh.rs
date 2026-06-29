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

/// One vertex of the tube mesh.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TubeVertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub tangent:  [f32; 3], // along the tube's length -- used by Anisotropic GGX shading
}

pub struct TubeMesh {
    pub vertices: Vec<TubeVertex>,
    pub indices:  Vec<u32>,
    pub bb_min:   Vec3,
    pub bb_max:   Vec3,
}

/// Skipped before sampling so the initial-condition transient (a stray
/// thread leading into the attractor) doesn't show up in the tube.
const WARMUP_STEPS: usize = 2_000;

fn build_centerline<A: Attractor>(attractor: &mut A, num_points: usize) -> Vec<Vec3> {
    for _ in 0..WARMUP_STEPS {
        if attractor.step().is_none() {
            break;
        }
    }
    let mut pts = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        match attractor.step() {
            Some(p) if p.pos.is_finite() => pts.push(p.pos),
            _ => break,
        }
    }
    pts
}

    pub fn build_centerline_cpu(config: &AttractorConfig, num_points: usize) -> Vec<Vec3> {
        
        match config.attractor_type {
            AttractorType::Lorenz => {
                let mut a = Lorenz::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Lorenz84 => {
                let mut a = Lorenz84::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Rossler => {
                let mut a = Rossler::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Thomas => {
                let mut a = Thomas::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ChaoticFlow => {
                let mut a = ChaoticFlow::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Pickover => {
                let mut a = Pickover::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Clifford => {
                let mut a = Clifford::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Icon => {
                let mut a = Icon::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::IconB => {
                let mut a = IconB::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::PolyA => {
                let mut a = PolyA::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::PolyAbs => {
                let mut a = PolyAbs::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::PolyB => {
                let mut a = PolyB::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::PolyC => {
                let mut a = PolyC::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::PolyPow => {
                let mut a = PolyPow::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::PolySin => {
                let mut a = PolySin::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::PolySprott => {
                let mut a = PolySprott::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::GenesioTesi => {
                let mut a = GenesioTesi::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Arneodo => {
                let mut a = Arneodo::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ChenCelikovsky => {
                let mut a = ChenCelikovsky::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ShimizuMorioka => {
                let mut a = ShimizuMorioka::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ThreeCellsCnn => {
                let mut a = ThreeCellsCnn::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Rucklidge => {
                let mut a = Rucklidge::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::RayleighBenard => {
                let mut a = RayleighBenard::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::BurkeShaw => {
                let mut a = BurkeShaw::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Sakarya => {
                let mut a = Sakarya::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::StrizhakKawczynski => {
                let mut a = StrizhakKawczynski::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Bouali => {
                let mut a = Bouali::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Halvorsen => {
                let mut a = Halvorsen::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Aizawa => {
                let mut a = Aizawa::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::DequanLi => {
                let mut a = DequanLi::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Hadley => {
                let mut a = Hadley::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::NoseHoover => {
                let mut a = NoseHoover::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::NewtonLeipnik => {
                let mut a = NewtonLeipnik::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Finance => {
                let mut a = Finance::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ChenLee => {
                let mut a = ChenLee::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzB => {
                let mut a = SprottLinzB::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzC => {
                let mut a = SprottLinzC::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzD => {
                let mut a = SprottLinzD::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzE => {
                let mut a = SprottLinzE::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzF => {
                let mut a = SprottLinzF::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzG => {
                let mut a = SprottLinzG::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzH => {
                let mut a = SprottLinzH::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzI => {
                let mut a = SprottLinzI::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzJ => {
                let mut a = SprottLinzJ::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzK => {
                let mut a = SprottLinzK::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzL => {
                let mut a = SprottLinzL::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzM => {
                let mut a = SprottLinzM::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzN => {
                let mut a = SprottLinzN::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzO => {
                let mut a = SprottLinzO::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzP => {
                let mut a = SprottLinzP::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzQ => {
                let mut a = SprottLinzQ::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzR => {
                let mut a = SprottLinzR::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SprottLinzS => {
                let mut a = SprottLinzS::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Act => {
                let mut a = Act::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::AnishchenkoAstakhov => {
                let mut a = AnishchenkoAstakhov::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Arnold => {
                let mut a = Arnold::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::BoualiIii => {
                let mut a = BoualiIii::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Burgers => {
                let mut a = Burgers::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::BusinessCycleMap => {
                let mut a = BusinessCycleMap::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Cathala => {
                let mut a = Cathala::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ChuaCubic => {
                let mut a = ChuaCubic::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Coullet => {
                let mut a = Coullet::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Dadras => {
                let mut a = Dadras::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ElhadjSprott => {
                let mut a = ElhadjSprott::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ElhadjSprottA => {
                let mut a = ElhadjSprottA::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ElhadjSprottCMap => {
                let mut a = ElhadjSprottCMap::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::FourWing => {
                let mut a = FourWing::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::FourWing2 => {
                let mut a = FourWing2::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::FourWing3 => {
                let mut a = FourWing3::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Gingerbread => {
                let mut a = Gingerbread::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::GumowskiMira => {
                let mut a = GumowskiMira::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Hca => {
                let mut a = Hca::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::HeagyHammel => {
                let mut a = HeagyHammel::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Hopalong => {
                let mut a = Hopalong::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Ikeda => {
                let mut a = Ikeda::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Ikeda1 => {
                let mut a = Ikeda1::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::LiuChen => {
                let mut a = LiuChen::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::LorenzMod1 => {
                let mut a = LorenzMod1::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::LorenzMod2 => {
                let mut a = LorenzMod2::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::LueChen => {
                let mut a = LueChen::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Mira => {
                let mut a = Mira::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ModifiedLozi => {
                let mut a = ModifiedLozi::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::MultiChuaII => {
                let mut a = MultiChuaII::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::MultifoldHenon => {
                let mut a = MultifoldHenon::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Popcorn => {
                let mut a = Popcorn::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Popcorn2 => {
                let mut a = Popcorn2::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Qi3D => {
                let mut a = Qi3D::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::QiChen => {
                let mut a = QiChen::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Robinson => {
                let mut a = Robinson::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Serpentine => {
                let mut a = Serpentine::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::StrelkovaAnishchenko => {
                let mut a = StrelkovaAnishchenko::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Tsucs1 => {
                let mut a = Tsucs1::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Tsucs2 => {
                let mut a = Tsucs2::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::WangSun => {
                let mut a = WangSun::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::DeJong => {
                let mut a = DeJong::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::MacMillan => {
                let mut a = MacMillan::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::MarottoLorenz => {
                let mut a = MarottoLorenz::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::MaynardSmith => {
                let mut a = MaynardSmith::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::NishikawaKaneko => {
                let mut a = NishikawaKaneko::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::QHenon2 => {
                let mut a = QHenon2::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::RulkovMap => {
                let mut a = RulkovMap::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SineDelay => {
                let mut a = SineDelay::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::SineSineMap => {
                let mut a = SineSineMap::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Svensson => {
                let mut a = Svensson::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Tinkerbell => {
                let mut a = Tinkerbell::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::YangCao => {
                let mut a = YangCao::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::Zhou => {
                let mut a = Zhou::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
            AttractorType::ZhouChen => {
                let mut a = ZhouChen::new();
                a.reset(&config.params);
                build_centerline(&mut a, num_points)
            }
        }
    }

/// An orthonormal frame transported along the centerline without twisting,
/// via Hanson's "double reflection" rotation-minimizing-frame method. Unlike
/// a Frenet frame (which flips 180 degrees at curvature inflection/zero-
/// curvature points -- common in chaotic flows, e.g. near Lorenz's saddle),
/// this propagates each frame by the minimal rotation that carries the old
/// tangent onto the new one, so the cross-section never twists or jitters.
struct Frame {
    tangent: Vec3,
    normal:  Vec3,
}

fn seed_frame(tangent: Vec3) -> Frame {
    // Any "up" not parallel to the tangent; fall back to a second axis if
    // the trajectory's first segment happens to run along Y (e.g. a planar
    // map whose first two points are axis-aligned).
    let up = if tangent.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let normal = up.cross(tangent).normalize();
    Frame { tangent, normal }
}

fn transport_frame(prev: &Frame, new_tangent: Vec3, p_prev: Vec3, p_curr: Vec3) -> Frame {
    // Double reflection: reflect prev's normal across the plane bisecting
    // (p_curr - p_prev), then reflect again across the plane bisecting
    // (prev.tangent -> new_tangent), yielding a minimally-rotated frame.
    let v1 = p_curr - p_prev;
    let c1 = v1.dot(v1);
    if c1 < 1e-12 {
        return Frame { tangent: new_tangent, normal: prev.normal };
    }
    let r_l = prev.normal - v1 * (2.0 / c1) * v1.dot(prev.normal);
    let t_l = prev.tangent - v1 * (2.0 / c1) * v1.dot(prev.tangent);

    let v2 = new_tangent - t_l;
    let c2 = v2.dot(v2);
    let normal = if c2 < 1e-12 {
        r_l
    } else {
        r_l - v2 * (2.0 / c2) * v2.dot(r_l)
    };
    Frame { tangent: new_tangent, normal: normal.normalize() }
}

/// How many Catmull-Rom sub-segments to insert between each pair of raw
/// centerline points. Fast-moving regions of a flow can leave consecutive
/// `step()` samples far apart, which the rotation-minimizing sweep would
/// otherwise connect with a sharp angular joint; spline smoothing turns that
/// into a smooth curve instead, at the cost of multiplying the final vertex
/// count by roughly this factor.
const SPLINE_SUBDIVISIONS: u32 = 4;

/// Centripetal Catmull-Rom (knot spacing by sqrt of segment length) rather
/// than the uniform variant: chaotic flows sample wildly uneven segment
/// lengths (fast vs. slow regions), and uniform Catmull-Rom is known to
/// overshoot into loops/cusps on exactly that kind of unevenly-spaced input.
/// The centripetal parameterization stays well-behaved regardless of how
/// uneven the input spacing is.
fn catmull_rom_point(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    const ALPHA: f32 = 0.5;
    let d01 = (p1 - p0).length().max(1e-6).powf(ALPHA);
    let d12 = (p2 - p1).length().max(1e-6).powf(ALPHA);
    let d23 = (p3 - p2).length().max(1e-6).powf(ALPHA);

    let t0 = 0.0f32;
    let t1 = t0 + d01;
    let t2 = t1 + d12;
    let t3 = t2 + d23;
    let tt = t1 + t * (t2 - t1); // t in [0,1] maps onto the [t1, t2] knot span

    let a1 = p0 * ((t1 - tt) / (t1 - t0)) + p1 * ((tt - t0) / (t1 - t0));
    let a2 = p1 * ((t2 - tt) / (t2 - t1)) + p2 * ((tt - t1) / (t2 - t1));
    let a3 = p2 * ((t3 - tt) / (t3 - t2)) + p3 * ((tt - t2) / (t3 - t2));

    let b1 = a1 * ((t2 - tt) / (t2 - t0)) + a2 * ((tt - t0) / (t2 - t0));
    let b2 = a2 * ((t3 - tt) / (t3 - t1)) + a3 * ((tt - t1) / (t3 - t1));

    b1 * ((t2 - tt) / (t2 - t1)) + b2 * ((tt - t1) / (t2 - t1))
}

/// Resample a polyline through a uniform Catmull-Rom spline, replacing each
/// straight segment with `SPLINE_SUBDIVISIONS` smoothly-curved sub-segments
/// that still pass through every original point.
fn catmull_rom_smooth(points: &[Vec3]) -> Vec<Vec3> {
    let n = points.len();
    if n < 4 || SPLINE_SUBDIVISIONS <= 1 {
        return points.to_vec();
    }
    let mut out = Vec::with_capacity(n * SPLINE_SUBDIVISIONS as usize);
    for i in 0..n - 1 {
        let p0 = points[i.saturating_sub(1)];
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = points[(i + 2).min(n - 1)];
        let steps = if i == n - 2 { SPLINE_SUBDIVISIONS + 1 } else { SPLINE_SUBDIVISIONS };
        for s in 0..steps {
            let t = s as f32 / SPLINE_SUBDIVISIONS as f32;
            out.push(catmull_rom_point(p0, p1, p2, p3, t));
        }
    }
    out
}

/// Sweep an N-sided ring around `centerline` using rotation-minimizing
/// frames, producing a closed tube mesh (triangle list).
pub fn build_tube_mesh(centerline: &[Vec3], radius: f32, sides: u32) -> TubeMesh {
    let sides = sides.max(3);
    if centerline.len() < 2 {
        return TubeMesh { vertices: Vec::new(), indices: Vec::new(), bb_min: Vec3::ZERO, bb_max: Vec3::ZERO };
    }

    // Drop consecutive duplicate points (degenerate segments -- common right
    // after the orbit nearly revisits a prior point in slow-moving regions),
    // which would otherwise produce a zero-length tangent.
    let mut pts: Vec<Vec3> = Vec::with_capacity(centerline.len());
    for &p in centerline {
        if pts.last().map_or(true, |&last: &Vec3| (p - last).length_squared() > 1e-12) {
            pts.push(p);
        }
    }
    if pts.len() < 2 {
        return TubeMesh { vertices: Vec::new(), indices: Vec::new(), bb_min: Vec3::ZERO, bb_max: Vec3::ZERO };
    }
    let pts = catmull_rom_smooth(&pts);

    let mut tangents = Vec::with_capacity(pts.len());
    for i in 0..pts.len() {
        let t = if i == 0 {
            (pts[1] - pts[0]).normalize()
        } else if i == pts.len() - 1 {
            (pts[i] - pts[i - 1]).normalize()
        } else {
            ((pts[i + 1] - pts[i - 1])).normalize()
        };
        tangents.push(t);
    }

    let mut frames = Vec::with_capacity(pts.len());
    frames.push(seed_frame(tangents[0]));
    for i in 1..pts.len() {
        let prev = &frames[i - 1];
        frames.push(transport_frame(prev, tangents[i], pts[i - 1], pts[i]));
    }

    let mut bb_min = Vec3::splat(f32::MAX);
    let mut bb_max = Vec3::splat(f32::MIN);
    let mut vertices = Vec::with_capacity(pts.len() * sides as usize);
    for i in 0..pts.len() {
        let frame = &frames[i];
        let binormal = frame.tangent.cross(frame.normal).normalize();
        for k in 0..sides {
            let theta = std::f32::consts::TAU * (k as f32) / (sides as f32);
            let radial = frame.normal * theta.cos() + binormal * theta.sin();
            let pos = pts[i] + radial * radius;
            bb_min = bb_min.min(pos);
            bb_max = bb_max.max(pos);
            vertices.push(TubeVertex {
                position: pos.to_array(),
                normal:   radial.to_array(),
                tangent:  frame.tangent.to_array(),
            });
        }
    }

    let mut indices = Vec::with_capacity((pts.len() - 1) * sides as usize * 6);
    for i in 0..pts.len() - 1 {
        let ring0 = i as u32 * sides;
        let ring1 = (i + 1) as u32 * sides;
        for k in 0..sides {
            let k_next = (k + 1) % sides;
            let a = ring0 + k;
            let b = ring0 + k_next;
            let c = ring1 + k;
            let d = ring1 + k_next;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }

    TubeMesh { vertices, indices, bb_min, bb_max }
}

/// Build the centerline + tube mesh for the given config in one call -- the
/// entry point `main.rs` invokes whenever the Solid-mode mesh needs rebuilding.
pub fn build_solid_mesh(config: &AttractorConfig, num_points: usize, radius: f32, sides: u32) -> TubeMesh {
    let centerline = build_centerline_cpu(config, num_points);
    build_tube_mesh(&centerline, radius, sides)
}
