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
    GenesioTesi,
    Arneodo,
    ChenCelikovsky,
    ShimizuMorioka,
    ThreeCellsCnn,
    Rucklidge,
    RayleighBenard,
    BurkeShaw,
    Sakarya,
    StrizhakKawczynski,
    Bouali,
    Halvorsen,
    Aizawa,
    DequanLi,
    Hadley,
    NoseHoover,
    NewtonLeipnik,
    Finance,
    ChenLee,
    SprottLinzB,
    SprottLinzC,
    SprottLinzD,
    SprottLinzE,
    SprottLinzF,
    SprottLinzG,
    SprottLinzH,
    SprottLinzI,
    SprottLinzJ,
    SprottLinzK,
    SprottLinzL,
    SprottLinzM,
    SprottLinzN,
    SprottLinzO,
    SprottLinzP,
    SprottLinzQ,
    SprottLinzR,
    SprottLinzS,
    Act,
    AnishchenkoAstakhov,
    Arnold,
    BoualiIii,
    Burgers,
    BusinessCycleMap,
    Cathala,
    ChuaCubic,
    Coullet,
    Dadras,
    ElhadjSprott,
    ElhadjSprottA,
    ElhadjSprottCMap,
    FourWing,
    FourWing2,
    FourWing3,
    Gingerbread,
    GumowskiMira,
    Hca,
    HeagyHammel,
    Hopalong,
    Ikeda,
    Ikeda1,
    LiuChen,
    LorenzMod1,
    LorenzMod2,
    LueChen,
    Mira,
    ModifiedLozi,
    MultiChuaII,
    MultifoldHenon,
    Popcorn,
    Popcorn2,
    Qi3D,
    QiChen,
    Robinson,
    Serpentine,
    StrelkovaAnishchenko,
    Tsucs1,
    Tsucs2,
    WangSun,
    DeJong,
    MacMillan,
    MarottoLorenz,
    MaynardSmith,
    NishikawaKaneko,
    QHenon2,
    RulkovMap,
    SineDelay,
    SineSineMap,
    Svensson,
    Tinkerbell,
    YangCao,
    Zhou,
    ZhouChen,
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
            Self::GenesioTesi => "Genesio-Tesi",
            Self::Arneodo     => "Arneodo",
            Self::ChenCelikovsky    => "Chen-Celikovsky",
            Self::ShimizuMorioka    => "Shimizu-Morioka",
            Self::ThreeCellsCnn     => "3-Cells CNN",
            Self::Rucklidge         => "Rucklidge",
            Self::RayleighBenard    => "Rayleigh-Bénard",
            Self::BurkeShaw         => "Burke-Shaw",
            Self::Sakarya           => "Sakarya",
            Self::StrizhakKawczynski => "Strizhak-Kawczynski",
            Self::Bouali            => "Bouali",
            Self::Halvorsen         => "Halvorsen",
            Self::Aizawa            => "Aizawa",
            Self::DequanLi          => "Dequan Li",
            Self::Hadley            => "Hadley",
            Self::NoseHoover        => "Nosé-Hoover",
            Self::NewtonLeipnik     => "Newton-Leipnik",
            Self::Finance           => "Finance",
            Self::ChenLee           => "Chen-Lee",
            Self::SprottLinzB => "Sprott-Linz B",
            Self::SprottLinzC => "Sprott-Linz C",
            Self::SprottLinzD => "Sprott-Linz D",
            Self::SprottLinzE => "Sprott-Linz E",
            Self::SprottLinzF => "Sprott-Linz F",
            Self::SprottLinzG => "Sprott-Linz G",
            Self::SprottLinzH => "Sprott-Linz H",
            Self::SprottLinzI => "Sprott-Linz I",
            Self::SprottLinzJ => "Sprott-Linz J",
            Self::SprottLinzK => "Sprott-Linz K",
            Self::SprottLinzL => "Sprott-Linz L",
            Self::SprottLinzM => "Sprott-Linz M",
            Self::SprottLinzN => "Sprott-Linz N",
            Self::SprottLinzO => "Sprott-Linz O",
            Self::SprottLinzP => "Sprott-Linz P",
            Self::SprottLinzQ => "Sprott-Linz Q",
            Self::SprottLinzR => "Sprott-Linz R",
            Self::SprottLinzS => "Sprott-Linz S",
            Self::Act => "ACT",
            Self::AnishchenkoAstakhov => "Anishchenko-Astakhov",
            Self::Arnold => "Arnold",
            Self::BoualiIii => "Bouali III",
            Self::Burgers => "Burgers",
            Self::BusinessCycleMap => "Business Cycle Map",
            Self::Cathala => "Cathala",
            Self::ChuaCubic => "Chua Cubic",
            Self::Coullet => "Coullet",
            Self::Dadras => "Dadras",
            Self::ElhadjSprott => "Elhadj-Sprott",
            Self::ElhadjSprottA => "Elhadj-Sprott A",
            Self::ElhadjSprottCMap => "Elhadj-Sprott C",
            Self::FourWing => "Four-Wing",
            Self::FourWing2 => "Four-Wing 2",
            Self::FourWing3 => "Four-Wing 3",
            Self::Gingerbread => "Gingerbread",
            Self::GumowskiMira => "Gumowski-Mira",
            Self::Hca => "HCA",
            Self::HeagyHammel => "Heagy-Hammel",
            Self::Hopalong => "Hopalong",
            Self::Ikeda => "Ikeda",
            Self::Ikeda1 => "Ikeda (Variant)",
            Self::LiuChen => "Liu-Chen",
            Self::LorenzMod1 => "Lorenz Mod 1",
            Self::LorenzMod2 => "Lorenz Mod 2",
            Self::LueChen => "Lue-Chen",
            Self::Mira => "Mira",
            Self::ModifiedLozi => "Modified Lozi",
            Self::MultiChuaII => "Multi-Chua II",
            Self::MultifoldHenon => "Multifold Henon",
            Self::Popcorn => "Popcorn",
            Self::Popcorn2 => "Popcorn 2",
            Self::Qi3D => "Qi 3D",
            Self::QiChen => "Qi-Chen",
            Self::Robinson => "Robinson",
            Self::Serpentine => "Serpentine",
            Self::StrelkovaAnishchenko => "Strelkova-Anishchenko",
            Self::Tsucs1 => "TSUCS-1",
            Self::Tsucs2 => "TSUCS-2",
            Self::WangSun => "Wang-Sun",
            Self::DeJong => "Peter de Jong",
            Self::MacMillan => "MacMillan",
            Self::MarottoLorenz => "Marotto-Lorenz",
            Self::MaynardSmith => "Maynard Smith",
            Self::NishikawaKaneko => "Nishikawa-Kaneko",
            Self::QHenon2 => "Q-Henon 2",
            Self::RulkovMap => "Rulkov Map",
            Self::SineDelay => "Sine-Delay",
            Self::SineSineMap => "Sine Sine Map",
            Self::Svensson => "Svensson",
            Self::Tinkerbell => "Tinkerbell",
            Self::YangCao => "Yang-Cao",
            Self::Zhou => "Zhou",
            Self::ZhouChen => "Zhou-Chen",
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
            Self::GenesioTesi => GenesioTesi::param_descriptors(),
            Self::Arneodo     => Arneodo::param_descriptors(),
            Self::ChenCelikovsky     => ChenCelikovsky::param_descriptors(),
            Self::ShimizuMorioka     => ShimizuMorioka::param_descriptors(),
            Self::ThreeCellsCnn      => ThreeCellsCnn::param_descriptors(),
            Self::Rucklidge          => Rucklidge::param_descriptors(),
            Self::RayleighBenard     => RayleighBenard::param_descriptors(),
            Self::BurkeShaw          => BurkeShaw::param_descriptors(),
            Self::Sakarya            => Sakarya::param_descriptors(),
            Self::StrizhakKawczynski => StrizhakKawczynski::param_descriptors(),
            Self::Bouali             => Bouali::param_descriptors(),
            Self::Halvorsen          => Halvorsen::param_descriptors(),
            Self::Aizawa             => Aizawa::param_descriptors(),
            Self::DequanLi           => DequanLi::param_descriptors(),
            Self::Hadley             => Hadley::param_descriptors(),
            Self::NoseHoover         => NoseHoover::param_descriptors(),
            Self::NewtonLeipnik      => NewtonLeipnik::param_descriptors(),
            Self::Finance            => Finance::param_descriptors(),
            Self::ChenLee            => ChenLee::param_descriptors(),
            Self::SprottLinzB => SprottLinzB::param_descriptors(),
            Self::SprottLinzC => SprottLinzC::param_descriptors(),
            Self::SprottLinzD => SprottLinzD::param_descriptors(),
            Self::SprottLinzE => SprottLinzE::param_descriptors(),
            Self::SprottLinzF => SprottLinzF::param_descriptors(),
            Self::SprottLinzG => SprottLinzG::param_descriptors(),
            Self::SprottLinzH => SprottLinzH::param_descriptors(),
            Self::SprottLinzI => SprottLinzI::param_descriptors(),
            Self::SprottLinzJ => SprottLinzJ::param_descriptors(),
            Self::SprottLinzK => SprottLinzK::param_descriptors(),
            Self::SprottLinzL => SprottLinzL::param_descriptors(),
            Self::SprottLinzM => SprottLinzM::param_descriptors(),
            Self::SprottLinzN => SprottLinzN::param_descriptors(),
            Self::SprottLinzO => SprottLinzO::param_descriptors(),
            Self::SprottLinzP => SprottLinzP::param_descriptors(),
            Self::SprottLinzQ => SprottLinzQ::param_descriptors(),
            Self::SprottLinzR => SprottLinzR::param_descriptors(),
            Self::SprottLinzS => SprottLinzS::param_descriptors(),
            Self::Act => Act::param_descriptors(),
            Self::AnishchenkoAstakhov => AnishchenkoAstakhov::param_descriptors(),
            Self::Arnold => Arnold::param_descriptors(),
            Self::BoualiIii => BoualiIii::param_descriptors(),
            Self::Burgers => Burgers::param_descriptors(),
            Self::BusinessCycleMap => BusinessCycleMap::param_descriptors(),
            Self::Cathala => Cathala::param_descriptors(),
            Self::ChuaCubic => ChuaCubic::param_descriptors(),
            Self::Coullet => Coullet::param_descriptors(),
            Self::Dadras => Dadras::param_descriptors(),
            Self::ElhadjSprott => ElhadjSprott::param_descriptors(),
            Self::ElhadjSprottA => ElhadjSprottA::param_descriptors(),
            Self::ElhadjSprottCMap => ElhadjSprottCMap::param_descriptors(),
            Self::FourWing => FourWing::param_descriptors(),
            Self::FourWing2 => FourWing2::param_descriptors(),
            Self::FourWing3 => FourWing3::param_descriptors(),
            Self::Gingerbread => Gingerbread::param_descriptors(),
            Self::GumowskiMira => GumowskiMira::param_descriptors(),
            Self::Hca => Hca::param_descriptors(),
            Self::HeagyHammel => HeagyHammel::param_descriptors(),
            Self::Hopalong => Hopalong::param_descriptors(),
            Self::Ikeda => Ikeda::param_descriptors(),
            Self::Ikeda1 => Ikeda1::param_descriptors(),
            Self::LiuChen => LiuChen::param_descriptors(),
            Self::LorenzMod1 => LorenzMod1::param_descriptors(),
            Self::LorenzMod2 => LorenzMod2::param_descriptors(),
            Self::LueChen => LueChen::param_descriptors(),
            Self::Mira => Mira::param_descriptors(),
            Self::ModifiedLozi => ModifiedLozi::param_descriptors(),
            Self::MultiChuaII => MultiChuaII::param_descriptors(),
            Self::MultifoldHenon => MultifoldHenon::param_descriptors(),
            Self::Popcorn => Popcorn::param_descriptors(),
            Self::Popcorn2 => Popcorn2::param_descriptors(),
            Self::Qi3D => Qi3D::param_descriptors(),
            Self::QiChen => QiChen::param_descriptors(),
            Self::Robinson => Robinson::param_descriptors(),
            Self::Serpentine => Serpentine::param_descriptors(),
            Self::StrelkovaAnishchenko => StrelkovaAnishchenko::param_descriptors(),
            Self::Tsucs1 => Tsucs1::param_descriptors(),
            Self::Tsucs2 => Tsucs2::param_descriptors(),
            Self::WangSun => WangSun::param_descriptors(),
            Self::DeJong => DeJong::param_descriptors(),
            Self::MacMillan => MacMillan::param_descriptors(),
            Self::MarottoLorenz => MarottoLorenz::param_descriptors(),
            Self::MaynardSmith => MaynardSmith::param_descriptors(),
            Self::NishikawaKaneko => NishikawaKaneko::param_descriptors(),
            Self::QHenon2 => QHenon2::param_descriptors(),
            Self::RulkovMap => RulkovMap::param_descriptors(),
            Self::SineDelay => SineDelay::param_descriptors(),
            Self::SineSineMap => SineSineMap::param_descriptors(),
            Self::Svensson => Svensson::param_descriptors(),
            Self::Tinkerbell => Tinkerbell::param_descriptors(),
            Self::YangCao => YangCao::param_descriptors(),
            Self::Zhou => Zhou::param_descriptors(),
            Self::ZhouChen => ZhouChen::param_descriptors(),
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
            | Self::GenesioTesi | Self::Arneodo | Self::ChenCelikovsky | Self::ShimizuMorioka
            | Self::ThreeCellsCnn | Self::Rucklidge | Self::RayleighBenard | Self::BurkeShaw
            | Self::Sakarya | Self::StrizhakKawczynski | Self::Bouali | Self::Halvorsen
            | Self::Aizawa | Self::DequanLi | Self::Hadley | Self::NoseHoover
            | Self::NewtonLeipnik | Self::Finance | Self::ChenLee
            | Self::SprottLinzB | Self::SprottLinzC | Self::SprottLinzD
            | Self::SprottLinzE | Self::SprottLinzF | Self::SprottLinzG | Self::SprottLinzH
            | Self::SprottLinzI | Self::SprottLinzJ | Self::SprottLinzK | Self::SprottLinzL
            | Self::SprottLinzM | Self::SprottLinzN | Self::SprottLinzO | Self::SprottLinzP
            | Self::SprottLinzQ | Self::SprottLinzR | Self::SprottLinzS
            | Self::Act
            | Self::AnishchenkoAstakhov
            | Self::BoualiIii
            | Self::ChuaCubic
            | Self::Coullet
            | Self::Dadras
            | Self::ElhadjSprott
            | Self::FourWing
            | Self::FourWing2
            | Self::FourWing3
            | Self::LiuChen
            | Self::LorenzMod1
            | Self::LorenzMod2
            | Self::LueChen
            | Self::MultiChuaII
            | Self::Qi3D
            | Self::QiChen
            | Self::Robinson
            | Self::Tsucs1
            | Self::Tsucs2
            | Self::WangSun
            | Self::Zhou
            | Self::ZhouChen
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
        Self::GenesioTesi,
        Self::Arneodo,
        Self::ChenCelikovsky,
        Self::ShimizuMorioka,
        Self::ThreeCellsCnn,
        Self::Rucklidge,
        Self::RayleighBenard,
        Self::BurkeShaw,
        Self::Sakarya,
        Self::StrizhakKawczynski,
        Self::Bouali,
        Self::Halvorsen,
        Self::Aizawa,
        Self::DequanLi,
        Self::Hadley,
        Self::NoseHoover,
        Self::NewtonLeipnik,
        Self::Finance,
        Self::ChenLee,
        Self::SprottLinzB,
        Self::SprottLinzC,
        Self::SprottLinzD,
        Self::SprottLinzE,
        Self::SprottLinzF,
        Self::SprottLinzG,
        Self::SprottLinzH,
        Self::SprottLinzI,
        Self::SprottLinzJ,
        Self::SprottLinzK,
        Self::SprottLinzL,
        Self::SprottLinzM,
        Self::SprottLinzN,
        Self::SprottLinzO,
        Self::SprottLinzP,
        Self::SprottLinzQ,
        Self::SprottLinzR,
        Self::SprottLinzS,
        Self::Act,
        Self::AnishchenkoAstakhov,
        Self::Arnold,
        Self::BoualiIii,
        Self::Burgers,
        Self::BusinessCycleMap,
        Self::Cathala,
        Self::ChuaCubic,
        Self::Coullet,
        Self::Dadras,
        Self::ElhadjSprott,
        Self::ElhadjSprottA,
        Self::ElhadjSprottCMap,
        Self::FourWing,
        Self::FourWing2,
        Self::FourWing3,
        Self::Gingerbread,
        Self::GumowskiMira,
        Self::Hca,
        Self::HeagyHammel,
        Self::Hopalong,
        Self::Ikeda,
        Self::Ikeda1,
        Self::LiuChen,
        Self::LorenzMod1,
        Self::LorenzMod2,
        Self::LueChen,
        Self::Mira,
        Self::ModifiedLozi,
        Self::MultiChuaII,
        Self::MultifoldHenon,
        Self::Popcorn,
        Self::Popcorn2,
        Self::Qi3D,
        Self::QiChen,
        Self::Robinson,
        Self::Serpentine,
        Self::StrelkovaAnishchenko,
        Self::Tsucs1,
        Self::Tsucs2,
        Self::WangSun,
        Self::DeJong,
        Self::MacMillan,
        Self::MarottoLorenz,
        Self::MaynardSmith,
        Self::NishikawaKaneko,
        Self::QHenon2,
        Self::RulkovMap,
        Self::SineDelay,
        Self::SineSineMap,
        Self::Svensson,
        Self::Tinkerbell,
        Self::YangCao,
        Self::Zhou,
        Self::ZhouChen,
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
            AttractorType::GenesioTesi => {
                let mut a = GenesioTesi::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Arneodo => {
                let mut a = Arneodo::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ChenCelikovsky => {
                let mut a = ChenCelikovsky::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ShimizuMorioka => {
                let mut a = ShimizuMorioka::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ThreeCellsCnn => {
                let mut a = ThreeCellsCnn::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Rucklidge => {
                let mut a = Rucklidge::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::RayleighBenard => {
                let mut a = RayleighBenard::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::BurkeShaw => {
                let mut a = BurkeShaw::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Sakarya => {
                let mut a = Sakarya::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::StrizhakKawczynski => {
                let mut a = StrizhakKawczynski::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Bouali => {
                let mut a = Bouali::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Halvorsen => {
                let mut a = Halvorsen::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Aizawa => {
                let mut a = Aizawa::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::DequanLi => {
                let mut a = DequanLi::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Hadley => {
                let mut a = Hadley::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::NoseHoover => {
                let mut a = NoseHoover::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::NewtonLeipnik => {
                let mut a = NewtonLeipnik::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Finance => {
                let mut a = Finance::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ChenLee => {
                let mut a = ChenLee::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzB => {
                let mut a = SprottLinzB::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzC => {
                let mut a = SprottLinzC::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzD => {
                let mut a = SprottLinzD::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzE => {
                let mut a = SprottLinzE::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzF => {
                let mut a = SprottLinzF::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzG => {
                let mut a = SprottLinzG::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzH => {
                let mut a = SprottLinzH::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzI => {
                let mut a = SprottLinzI::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzJ => {
                let mut a = SprottLinzJ::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzK => {
                let mut a = SprottLinzK::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzL => {
                let mut a = SprottLinzL::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzM => {
                let mut a = SprottLinzM::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzN => {
                let mut a = SprottLinzN::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzO => {
                let mut a = SprottLinzO::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzP => {
                let mut a = SprottLinzP::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzQ => {
                let mut a = SprottLinzQ::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzR => {
                let mut a = SprottLinzR::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SprottLinzS => {
                let mut a = SprottLinzS::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Act => {
                let mut a = Act::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::AnishchenkoAstakhov => {
                let mut a = AnishchenkoAstakhov::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Arnold => {
                let mut a = Arnold::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::BoualiIii => {
                let mut a = BoualiIii::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Burgers => {
                let mut a = Burgers::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::BusinessCycleMap => {
                let mut a = BusinessCycleMap::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Cathala => {
                let mut a = Cathala::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ChuaCubic => {
                let mut a = ChuaCubic::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Coullet => {
                let mut a = Coullet::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Dadras => {
                let mut a = Dadras::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ElhadjSprott => {
                let mut a = ElhadjSprott::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ElhadjSprottA => {
                let mut a = ElhadjSprottA::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ElhadjSprottCMap => {
                let mut a = ElhadjSprottCMap::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::FourWing => {
                let mut a = FourWing::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::FourWing2 => {
                let mut a = FourWing2::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::FourWing3 => {
                let mut a = FourWing3::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Gingerbread => {
                let mut a = Gingerbread::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::GumowskiMira => {
                let mut a = GumowskiMira::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Hca => {
                let mut a = Hca::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::HeagyHammel => {
                let mut a = HeagyHammel::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Hopalong => {
                let mut a = Hopalong::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Ikeda => {
                let mut a = Ikeda::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Ikeda1 => {
                let mut a = Ikeda1::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::LiuChen => {
                let mut a = LiuChen::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::LorenzMod1 => {
                let mut a = LorenzMod1::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::LorenzMod2 => {
                let mut a = LorenzMod2::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::LueChen => {
                let mut a = LueChen::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Mira => {
                let mut a = Mira::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ModifiedLozi => {
                let mut a = ModifiedLozi::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::MultiChuaII => {
                let mut a = MultiChuaII::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::MultifoldHenon => {
                let mut a = MultifoldHenon::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Popcorn => {
                let mut a = Popcorn::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Popcorn2 => {
                let mut a = Popcorn2::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Qi3D => {
                let mut a = Qi3D::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::QiChen => {
                let mut a = QiChen::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Robinson => {
                let mut a = Robinson::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Serpentine => {
                let mut a = Serpentine::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::StrelkovaAnishchenko => {
                let mut a = StrelkovaAnishchenko::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Tsucs1 => {
                let mut a = Tsucs1::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Tsucs2 => {
                let mut a = Tsucs2::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::WangSun => {
                let mut a = WangSun::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::DeJong => {
                let mut a = DeJong::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::MacMillan => {
                let mut a = MacMillan::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::MarottoLorenz => {
                let mut a = MarottoLorenz::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::MaynardSmith => {
                let mut a = MaynardSmith::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::NishikawaKaneko => {
                let mut a = NishikawaKaneko::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::QHenon2 => {
                let mut a = QHenon2::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::RulkovMap => {
                let mut a = RulkovMap::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SineDelay => {
                let mut a = SineDelay::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::SineSineMap => {
                let mut a = SineSineMap::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Svensson => {
                let mut a = Svensson::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Tinkerbell => {
                let mut a = Tinkerbell::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::YangCao => {
                let mut a = YangCao::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::Zhou => {
                let mut a = Zhou::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
            AttractorType::ZhouChen => {
                let mut a = ZhouChen::new();
                a.reset(&self.params);
                estimate_bounds(&mut a, 200_000)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEW_TYPES: &[AttractorType] = &[
        AttractorType::GenesioTesi,
        AttractorType::Arneodo,
        AttractorType::ChenCelikovsky,
        AttractorType::ShimizuMorioka,
        AttractorType::ThreeCellsCnn,
        AttractorType::Rucklidge,
        AttractorType::RayleighBenard,
        AttractorType::BurkeShaw,
        AttractorType::Sakarya,
        AttractorType::StrizhakKawczynski,
        AttractorType::Bouali,
        AttractorType::Halvorsen,
        AttractorType::Aizawa,
        AttractorType::DequanLi,
        AttractorType::Hadley,
        AttractorType::NoseHoover,
        AttractorType::NewtonLeipnik,
        AttractorType::Finance,
        AttractorType::ChenLee,
        AttractorType::SprottLinzB,
        AttractorType::SprottLinzC,
        AttractorType::SprottLinzD,
        AttractorType::SprottLinzE,
        AttractorType::SprottLinzF,
        AttractorType::SprottLinzG,
        AttractorType::SprottLinzH,
        AttractorType::SprottLinzI,
        AttractorType::SprottLinzJ,
        AttractorType::SprottLinzK,
        AttractorType::SprottLinzL,
        AttractorType::SprottLinzM,
        AttractorType::SprottLinzN,
        AttractorType::SprottLinzO,
        AttractorType::SprottLinzP,
        AttractorType::SprottLinzQ,
        AttractorType::SprottLinzR,
        AttractorType::SprottLinzS,
        AttractorType::Act,
        AttractorType::AnishchenkoAstakhov,
        AttractorType::Arnold,
        AttractorType::BoualiIii,
        AttractorType::Burgers,
        AttractorType::BusinessCycleMap,
        AttractorType::Cathala,
        AttractorType::ChuaCubic,
        AttractorType::Coullet,
        AttractorType::Dadras,
        AttractorType::ElhadjSprott,
        AttractorType::ElhadjSprottA,
        AttractorType::ElhadjSprottCMap,
        AttractorType::FourWing,
        AttractorType::FourWing2,
        AttractorType::FourWing3,
        AttractorType::Gingerbread,
        AttractorType::GumowskiMira,
        AttractorType::Hca,
        AttractorType::HeagyHammel,
        AttractorType::Hopalong,
        AttractorType::Ikeda,
        AttractorType::Ikeda1,
        AttractorType::LiuChen,
        AttractorType::LorenzMod1,
        AttractorType::LorenzMod2,
        AttractorType::LueChen,
        AttractorType::Mira,
        AttractorType::ModifiedLozi,
        AttractorType::MultiChuaII,
        AttractorType::MultifoldHenon,
        AttractorType::Popcorn,
        AttractorType::Popcorn2,
        AttractorType::Qi3D,
        AttractorType::QiChen,
        AttractorType::Robinson,
        AttractorType::Serpentine,
        AttractorType::StrelkovaAnishchenko,
        AttractorType::Tsucs1,
        AttractorType::Tsucs2,
        AttractorType::WangSun,
        AttractorType::DeJong,
        AttractorType::MacMillan,
        AttractorType::MarottoLorenz,
        AttractorType::MaynardSmith,
        AttractorType::NishikawaKaneko,
        AttractorType::QHenon2,
        AttractorType::RulkovMap,
        AttractorType::SineDelay,
        AttractorType::SineSineMap,
        AttractorType::Svensson,
        AttractorType::Tinkerbell,
        AttractorType::YangCao,
        AttractorType::Zhou,
        AttractorType::ZhouChen,
    ];

    /// Every newly-ported attractor should settle onto a bounded, non-degenerate
    /// attractor from its default parameters — i.e. `estimate_bounds_cpu` (the
    /// exact function used to fit the camera on type switch) must return a
    /// finite bounding box with some actual spread, not a collapsed point or
    /// the diverged-trajectory fallback cube.
    #[test]
    fn new_attractors_settle_onto_a_bounded_orbit() {
        for &t in NEW_TYPES {
            let config = AttractorConfig::new(t);
            let (bb_min, bb_max) = config.estimate_bounds_cpu();
            assert!(bb_min.is_finite() && bb_max.is_finite(), "{:?}: non-finite bbox", t);
            let size = (bb_max - bb_min).length();
            assert!(size > 0.01, "{:?}: degenerate bbox (size {})", t, size);
        }
    }
}
