use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use glam::Mat4;
use sim::{AttractorConfig, AttractorType};

/// Interleaved u32 pairs: [density, speed_fixed] per pixel.
/// density (index 0) accumulates bilinear weights; speed (index 1) accumulates
/// log-encoded speed × weight / SPEED_SCALE — see sim.wgsl for details.
const ACCUM_ELEM_SIZE: u64 = 8;

const DEFAULT_SS_SCALE: u32 = 2;

/// Number of independent Lorenz trajectories simulated in parallel on the GPU.
const SIM_NUM_TRAJECTORIES: u32 = 8_192;
/// Euler steps each trajectory advances per compute dispatch (≈ one frame).
const SIM_STEPS_PER_DISPATCH: u32 = 512;

/// Composite HDR intermediate (luma before swapchain blit).
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Horizontal DE pass intermediate (raw blurred density, one f32 per pixel).
const DE_H_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Width (in texels) of the 1D gradient textures (256×1 Rgba8Unorm).
const GRADIENT_TEX_WIDTH: u32 = 256;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which compositing mode to use when rendering.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum RenderMode {
    /// Monochrome log-density with DE blur (original mode).
    #[default]
    Monochrome,
    /// Two-gradient blend: Gradient A by density, Gradient B by mean speed.
    Light,
}

/// How Gradient A (density) and Gradient B (speed) are combined in Light mode.
/// Numeric value is written directly into `CompositeParams::blend_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
#[repr(u32)]
pub enum BlendMode {
    #[default]
    Add      = 0,
    Subtract = 1,
    Multiply = 2,
    Divide   = 3,
    Lighten  = 4,
    Darken   = 5,
    HardLight = 6,
    SoftLight = 7,
    HardMix  = 8,
}

impl BlendMode {
    pub fn as_u32(self) -> u32 { self as u32 }

    pub fn label(self) -> &'static str {
        match self {
            Self::Add      => "Add",
            Self::Subtract => "Subtract",
            Self::Multiply => "Multiply",
            Self::Divide   => "Divide",
            Self::Lighten  => "Lighten",
            Self::Darken   => "Darken",
            Self::HardLight => "Hard Light",
            Self::SoftLight => "Soft Light",
            Self::HardMix  => "Hard Mix",
        }
    }

    pub const ALL: &'static [Self] = &[
        Self::Add, Self::Subtract, Self::Multiply, Self::Divide,
        Self::Lighten, Self::Darken, Self::HardLight, Self::SoftLight, Self::HardMix,
    ];
}

// ---------------------------------------------------------------------------
// GPU structs (must match WGSL layouts exactly)
// ---------------------------------------------------------------------------

// SimParams layout must match the WGSL struct in all sim shaders.
// p is array<vec4<f32>, N> in WGSL (std140 requires vec4 stride = 4×f32).
// Most shaders declare N=12 (192 bytes of p → total 272 bytes).
// sim_poly_sprott.wgsl declares N=43 (688 bytes of p → total 768 bytes).
// Binding a 768-byte buffer to shaders that only declare 272 bytes is valid
// since min_binding_size is None — wgpu only checks the lower bound.
#[repr(C)]
#[derive(Copy, Clone)]
struct SimParams {
    view_proj: [f32; 16],  // offset 0,  64 bytes
    ss_width:  u32,        // offset 64
    ss_height: u32,        // offset 68
    steps:     u32,        // offset 72
    num_traj:  u32,        // offset 76
    p:         [f32; 172], // offset 80, 688 bytes → total 768 bytes
}
// Safety: repr(C), all fields are Pod/Zeroable, no padding (all fields 4-byte aligned).
unsafe impl bytemuck::Pod for SimParams {}
unsafe impl bytemuck::Zeroable for SimParams {}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositeParams {
    pub width:           u32,
    pub height:          u32,
    pub log_max_density: f32,
    pub brightness:      f32,
    pub gamma:           f32,
    pub ss_width:        u32,
    pub ss_height:       u32,
    pub max_sigma:       f32,
    pub min_sigma:       f32,
    pub ss_scale:        u32,
    pub blend_mode:      u32,  // BlendMode::as_u32(); only used by composite_light.wgsl
    pub _pad1:           u32,
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

pub struct Histogram {
    width:        u32,
    height:       u32,
    pub ss_scale: u32,

    // ---- accumulation buffer (ss_scale× super-sampled) ----
    accum_buf: wgpu::Buffer,

    // ---- GPU simulation (one pipeline per attractor type) ----
    sim_pipelines:   HashMap<AttractorType, wgpu::ComputePipeline>,
    sim_bind_layout: wgpu::BindGroupLayout,
    sim_state_buf:   wgpu::Buffer,
    sim_params_buf:  wgpu::Buffer,
    cached_sim_bg:   wgpu::BindGroup,

    // ---- horizontal DE pass (accum → de_h_texture) ----
    de_h_pipeline:     wgpu::RenderPipeline,
    de_h_bind_layout:  wgpu::BindGroupLayout,
    de_h_texture:      wgpu::Texture,
    de_h_texture_view: wgpu::TextureView,

    // ---- monochrome composite pass (de_h_texture + accum → hdr_texture) ----
    composite_pipeline:    wgpu::RenderPipeline,
    composite_bind_layout: wgpu::BindGroupLayout,
    composite_params_buf:  wgpu::Buffer,

    // ---- light mode composite pass ----
    composite_light_pipeline:    wgpu::RenderPipeline,
    composite_light_bind_layout: wgpu::BindGroupLayout,

    // ---- gradient textures (256×1 Rgba8Unorm, shared by light mode) ----
    gradient_a_texture:      wgpu::Texture,
    gradient_a_texture_view: wgpu::TextureView,
    gradient_b_texture:      wgpu::Texture,
    gradient_b_texture_view: wgpu::TextureView,
    gradient_sampler:        wgpu::Sampler,

    // ---- blit pass (hdr_texture → swapchain) ----
    blit_pipeline:    wgpu::RenderPipeline,
    blit_bind_layout: wgpu::BindGroupLayout,
    blit_sampler:     wgpu::Sampler,

    // ---- HDR intermediate ----
    hdr_texture:      wgpu::Texture,
    hdr_texture_view: wgpu::TextureView,

    // ---- GPU max-density readback ----
    max_density_buf:  wgpu::Buffer,
    max_readback_buf: wgpu::Buffer,
    max_map_ready:    Arc<AtomicBool>,
    max_map_inflight: bool,
    pub last_max_density: u32,

    // ---- cached bind groups ----
    cached_de_h_bg:            wgpu::BindGroup,
    cached_composite_bg:       wgpu::BindGroup,
    cached_composite_light_bg: wgpu::BindGroup,
    cached_blit_bg:            wgpu::BindGroup,

    // ---- attractor seed (a known-good point on the current attractor) ----
    // Stored in SimParams p[7].xyz so GPU shaders can use it as a reset IC.
    // Uses Cell for interior mutability so reset_sim_states can keep &self.
    attractor_seed: std::cell::Cell<[f32; 3]>,
}

impl Histogram {
    pub fn new(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        width:  u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        initial_config: &AttractorConfig,
    ) -> Self {
        let ss_scale = DEFAULT_SS_SCALE;
        let ss_w = width  * ss_scale;
        let ss_h = height * ss_scale;

        let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("accum_buf"),
            size:               ss_w as u64 * ss_h as u64 * ACCUM_ELEM_SIZE,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_density_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("max_density_buf"),
            size:               4,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_SRC
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("max_readback_buf"),
            size:               4,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- GPU simulation pipelines (one per attractor type, shared bind layout) ----
        let sim_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("sim_bgl"),
            entries: &[
                bgl_storage_rw(0, wgpu::ShaderStages::COMPUTE), // states
                bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE), // accum
                bgl_storage_rw(2, wgpu::ShaderStages::COMPUTE), // max_density
                bgl_uniform(3,    wgpu::ShaderStages::COMPUTE), // sim_params
            ],
        });

        let sim_pipelines = {
            let mut map = HashMap::new();
            map.insert(AttractorType::Lorenz,   make_sim_pipeline(device, include_str!("../shaders/sim.wgsl"),          &sim_bind_layout, "sim_lorenz"));
            map.insert(AttractorType::Lorenz84, make_sim_pipeline(device, include_str!("../shaders/sim_lorenz84.wgsl"), &sim_bind_layout, "sim_lorenz84"));
            map.insert(AttractorType::Rossler,  make_sim_pipeline(device, include_str!("../shaders/sim_rossler.wgsl"),  &sim_bind_layout, "sim_rossler"));
            map.insert(AttractorType::Thomas,      make_sim_pipeline(device, include_str!("../shaders/sim_thomas.wgsl"),        &sim_bind_layout, "sim_thomas"));
            map.insert(AttractorType::ChaoticFlow, make_sim_pipeline(device, include_str!("../shaders/sim_chaotic_flow.wgsl"), &sim_bind_layout, "sim_chaotic_flow"));
            map.insert(AttractorType::Pickover,    make_sim_pipeline(device, include_str!("../shaders/sim_pickover.wgsl"),     &sim_bind_layout, "sim_pickover"));
            map.insert(AttractorType::Clifford,    make_sim_pipeline(device, include_str!("../shaders/sim_clifford.wgsl"),     &sim_bind_layout, "sim_clifford"));
            map.insert(AttractorType::Icon,        make_sim_pipeline(device, include_str!("../shaders/sim_icon.wgsl"),         &sim_bind_layout, "sim_icon"));
            map.insert(AttractorType::IconB,       make_sim_pipeline(device, include_str!("../shaders/sim_icon_b.wgsl"),       &sim_bind_layout, "sim_icon_b"));
            map.insert(AttractorType::PolyA,       make_sim_pipeline(device, include_str!("../shaders/sim_poly_a.wgsl"),       &sim_bind_layout, "sim_poly_a"));
            map.insert(AttractorType::PolyAbs,     make_sim_pipeline(device, include_str!("../shaders/sim_poly_abs.wgsl"),     &sim_bind_layout, "sim_poly_abs"));
            map.insert(AttractorType::PolyB,       make_sim_pipeline(device, include_str!("../shaders/sim_poly_b.wgsl"),       &sim_bind_layout, "sim_poly_b"));
            map.insert(AttractorType::PolyC,       make_sim_pipeline(device, include_str!("../shaders/sim_poly_c.wgsl"),       &sim_bind_layout, "sim_poly_c"));
            map.insert(AttractorType::PolyPow,     make_sim_pipeline(device, include_str!("../shaders/sim_poly_pow.wgsl"),     &sim_bind_layout, "sim_poly_pow"));
            map.insert(AttractorType::PolySin,     make_sim_pipeline(device, include_str!("../shaders/sim_poly_sin.wgsl"),     &sim_bind_layout, "sim_poly_sin"));
            map.insert(AttractorType::PolySprott,  make_sim_pipeline(device, include_str!("../shaders/sim_poly_sprott.wgsl"),  &sim_bind_layout, "sim_poly_sprott"));
            map
        };

        let sim_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("sim_state_buf"),
            size:               SIM_NUM_TRAJECTORIES as u64 * 16, // vec4<f32> per trajectory
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("sim_params_buf"),
            size:               std::mem::size_of::<SimParams>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload initial trajectory states spread across the attractor.
        let initial_states = make_attractor_states(SIM_NUM_TRAJECTORIES, initial_config);
        let initial_seed = seed_from_states(&initial_states);
        queue.write_buffer(&sim_state_buf, 0, bytemuck::cast_slice(&initial_states));

        // ---- horizontal DE pipeline ----
        let de_h_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("de_h_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/de_h.wgsl").into()),
        });

        let de_h_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("de_h_bgl"),
            entries: &[
                bgl_storage_r(0, wgpu::ShaderStages::FRAGMENT),
                bgl_uniform(1,   wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let de_h_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("de_h_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("de_h_pl"),
                bind_group_layouts:   &[&de_h_bind_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module:              &de_h_shader,
                entry_point:         "vs_main",
                buffers:             &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &de_h_shader,
                entry_point: "fs_main",
                targets:     &[Some(wgpu::ColorTargetState {
                    format:     DE_H_FORMAT,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:     wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        let (de_h_texture, de_h_texture_view) = Self::make_de_h_texture(device, width, height);

        // ---- composite pipeline (monochrome) ----
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("composite_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });

        let composite_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("composite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        multisampled:   false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                bgl_storage_r(1, wgpu::ShaderStages::FRAGMENT),
                bgl_uniform(2,   wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("composite_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("composite_pl"),
                bind_group_layouts:   &[&composite_bind_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module:              &composite_shader,
                entry_point:         "vs_main",
                buffers:             &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &composite_shader,
                entry_point: "fs_main",
                targets:     &[Some(wgpu::ColorTargetState {
                    format:     HDR_FORMAT,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:     wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        let composite_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("composite_params_buf"),
            size:               std::mem::size_of::<CompositeParams>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- gradient textures (256×1 Rgba8Unorm) ----
        let (gradient_a_texture, gradient_a_texture_view) =
            Self::make_gradient_texture(device, "gradient_a");
        let (gradient_b_texture, gradient_b_texture_view) =
            Self::make_gradient_texture(device, "gradient_b");

        let gradient_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:            Some("gradient_sampler"),
            address_mode_u:   wgpu::AddressMode::ClampToEdge,
            address_mode_v:   wgpu::AddressMode::ClampToEdge,
            mag_filter:       wgpu::FilterMode::Linear,
            min_filter:       wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ---- composite pipeline (light mode) ----
        let composite_light_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("composite_light_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/composite_light.wgsl").into(),
            ),
        });

        let composite_light_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   Some("composite_light_bgl"),
                entries: &[
                    // de_h_tex (non-filterable float texture)
                    wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty:         wgpu::BindingType::Texture {
                            multisampled:   false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    bgl_storage_r(1, wgpu::ShaderStages::FRAGMENT), // accum
                    bgl_uniform(2,   wgpu::ShaderStages::FRAGMENT),  // params
                    // gradient_a (filterable)
                    wgpu::BindGroupLayoutEntry {
                        binding:    3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty:         wgpu::BindingType::Texture {
                            multisampled:   false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // gradient_b (filterable)
                    wgpu::BindGroupLayoutEntry {
                        binding:    4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty:         wgpu::BindingType::Texture {
                            multisampled:   false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // gradient_sampler
                    wgpu::BindGroupLayoutEntry {
                        binding:    5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let composite_light_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:  Some("composite_light_pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label:                Some("composite_light_pl"),
                    bind_group_layouts:   &[&composite_light_bind_layout],
                    push_constant_ranges: &[],
                })),
                vertex: wgpu::VertexState {
                    module:              &composite_light_shader,
                    entry_point:         "vs_main",
                    buffers:             &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &composite_light_shader,
                    entry_point: "fs_main",
                    targets:     &[Some(wgpu::ColorTargetState {
                        format:     HDR_FORMAT,
                        blend:      None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive:     wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
                multiview:     None,
                cache:         None,
            });

        // ---- blit pipeline ----
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("blit_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });

        let blit_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        multisampled:   false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("blit_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("blit_pl"),
                bind_group_layouts:   &[&blit_bind_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module:              &blit_shader,
                entry_point:         "vs_main",
                buffers:             &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &blit_shader,
                entry_point: "fs_main",
                targets:     &[Some(wgpu::ColorTargetState {
                    format:     surface_format,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:     wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("blit_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (hdr_texture, hdr_texture_view) = Self::make_hdr_texture(device, width, height);

        // ---- cached bind groups ----
        let cached_sim_bg = Self::make_sim_bg(
            device, &sim_bind_layout,
            &sim_state_buf, &accum_buf, &max_density_buf, &sim_params_buf,
        );
        let cached_de_h_bg = Self::make_de_h_bg(
            device, &de_h_bind_layout, &accum_buf, &composite_params_buf,
        );
        let cached_composite_bg = Self::make_composite_bg(
            device, &composite_bind_layout,
            &de_h_texture_view, &accum_buf, &composite_params_buf,
        );
        let cached_composite_light_bg = Self::make_composite_light_bg(
            device, &composite_light_bind_layout,
            &de_h_texture_view, &accum_buf, &composite_params_buf,
            &gradient_a_texture_view, &gradient_b_texture_view, &gradient_sampler,
        );
        let cached_blit_bg = Self::make_blit_bg(
            device, &blit_bind_layout, &hdr_texture_view, &blit_sampler,
        );

        Self {
            width,
            height,
            ss_scale,
            accum_buf,
            sim_pipelines,
            sim_bind_layout,
            sim_state_buf,
            sim_params_buf,
            de_h_pipeline,
            de_h_bind_layout,
            de_h_texture,
            de_h_texture_view,
            composite_pipeline,
            composite_bind_layout,
            composite_params_buf,
            composite_light_pipeline,
            composite_light_bind_layout,
            gradient_a_texture,
            gradient_a_texture_view,
            gradient_b_texture,
            gradient_b_texture_view,
            gradient_sampler,
            blit_pipeline,
            blit_bind_layout,
            blit_sampler,
            hdr_texture,
            hdr_texture_view,
            max_density_buf,
            max_readback_buf,
            max_map_ready: Arc::new(AtomicBool::new(false)),
            max_map_inflight: false,
            last_max_density: 1,
            cached_sim_bg,
            cached_de_h_bg,
            cached_composite_bg,
            cached_composite_light_bg,
            cached_blit_bg,
            attractor_seed: std::cell::Cell::new(initial_seed),
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width  = width;
        self.height = height;
        // Clamp in case the new dimensions make the current scale too large.
        self.ss_scale = self.ss_scale.min(Self::max_feasible_ss(device, width, height));

        let ss_w = width  * self.ss_scale;
        let ss_h = height * self.ss_scale;
        self.accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("accum_buf"),
            size:               ss_w as u64 * ss_h as u64 * ACCUM_ELEM_SIZE,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (de_h_texture, de_h_texture_view) = Self::make_de_h_texture(device, width, height);
        self.de_h_texture      = de_h_texture;
        self.de_h_texture_view = de_h_texture_view;

        let (hdr_texture, hdr_texture_view) = Self::make_hdr_texture(device, width, height);
        self.hdr_texture      = hdr_texture;
        self.hdr_texture_view = hdr_texture_view;

        self.cached_sim_bg = Self::make_sim_bg(
            device, &self.sim_bind_layout,
            &self.sim_state_buf, &self.accum_buf, &self.max_density_buf, &self.sim_params_buf,
        );
        self.cached_de_h_bg = Self::make_de_h_bg(
            device, &self.de_h_bind_layout, &self.accum_buf, &self.composite_params_buf,
        );
        self.cached_composite_bg = Self::make_composite_bg(
            device, &self.composite_bind_layout,
            &self.de_h_texture_view, &self.accum_buf, &self.composite_params_buf,
        );
        self.cached_composite_light_bg = Self::make_composite_light_bg(
            device, &self.composite_light_bind_layout,
            &self.de_h_texture_view, &self.accum_buf, &self.composite_params_buf,
            &self.gradient_a_texture_view, &self.gradient_b_texture_view, &self.gradient_sampler,
        );
        self.cached_blit_bg = Self::make_blit_bg(
            device, &self.blit_bind_layout, &self.hdr_texture_view, &self.blit_sampler,
        );
    }

    /// Largest supported SS scale ({1, 2, 4}) whose accum buffer fits within
    /// `device.limits().max_buffer_size` for the given display dimensions.
    pub fn max_feasible_ss(device: &wgpu::Device, width: u32, height: u32) -> u32 {
        let max_bytes = device.limits().max_buffer_size;
        let base = width as u64 * height as u64 * ACCUM_ELEM_SIZE;
        if base == 0 { return 4; }
        let max_ss_sq = max_bytes / base;
        if max_ss_sq >= 16 { 4 } else if max_ss_sq >= 4 { 2 } else { 1 }
    }

    pub fn set_ss_scale(&mut self, device: &wgpu::Device, ss_scale: u32) {
        self.ss_scale = ss_scale.max(1).min(Self::max_feasible_ss(device, self.width, self.height));
        let ss_w = self.width  * self.ss_scale;
        let ss_h = self.height * self.ss_scale;
        // Trajectory states don't need resetting: the sim params upload in
        // dispatch_sim() carries the new ss_width/ss_height, so the GPU will
        // immediately splat into the correct coordinates. The caller is responsible
        // for triggering a histogram clear (pending_clear = true in main.rs).

        self.accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("accum_buf"),
            size:               ss_w as u64 * ss_h as u64 * ACCUM_ELEM_SIZE,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.cached_sim_bg = Self::make_sim_bg(
            device, &self.sim_bind_layout,
            &self.sim_state_buf, &self.accum_buf, &self.max_density_buf, &self.sim_params_buf,
        );
        self.cached_de_h_bg = Self::make_de_h_bg(
            device, &self.de_h_bind_layout, &self.accum_buf, &self.composite_params_buf,
        );
        self.cached_composite_bg = Self::make_composite_bg(
            device, &self.composite_bind_layout,
            &self.de_h_texture_view, &self.accum_buf, &self.composite_params_buf,
        );
        self.cached_composite_light_bg = Self::make_composite_light_bg(
            device, &self.composite_light_bind_layout,
            &self.de_h_texture_view, &self.accum_buf, &self.composite_params_buf,
            &self.gradient_a_texture_view, &self.gradient_b_texture_view, &self.gradient_sampler,
        );
        self.last_max_density = 1;
    }

    // ---- gradient texture uploads ----

    /// Upload a 256×1 RGBA8 gradient to the density gradient (Gradient A).
    /// `rgba8` must be exactly 256 × 4 = 1024 bytes.
    pub fn upload_gradient_a(&self, queue: &wgpu::Queue, rgba8: &[u8]) {
        Self::upload_gradient_texture(queue, &self.gradient_a_texture, rgba8);
    }

    /// Upload a 256×1 RGBA8 gradient to the speed gradient (Gradient B).
    /// `rgba8` must be exactly 256 × 4 = 1024 bytes.
    pub fn upload_gradient_b(&self, queue: &wgpu::Queue, rgba8: &[u8]) {
        Self::upload_gradient_texture(queue, &self.gradient_b_texture, rgba8);
    }

    fn upload_gradient_texture(queue: &wgpu::Queue, texture: &wgpu::Texture, rgba8: &[u8]) {
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            rgba8,
            wgpu::ImageDataLayout {
                offset:         0,
                bytes_per_row:  Some(GRADIENT_TEX_WIDTH * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d { width: GRADIENT_TEX_WIDTH, height: 1, depth_or_array_layers: 1 },
        );
    }

    // ---- bind group helpers ----

    fn make_sim_bg(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        state_buf: &wgpu::Buffer, accum_buf: &wgpu::Buffer,
        max_density_buf: &wgpu::Buffer, params_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim_bg"), layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accum_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: max_density_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        })
    }

    fn make_de_h_bg(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        accum_buf: &wgpu::Buffer, params_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("de_h_bg"), layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: accum_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        })
    }

    fn make_composite_bg(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        de_h_view: &wgpu::TextureView, accum_buf: &wgpu::Buffer, params_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bg"), layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(de_h_view),
                },
                wgpu::BindGroupEntry { binding: 1, resource: accum_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        })
    }

    fn make_composite_light_bg(
        device:           &wgpu::Device,
        layout:           &wgpu::BindGroupLayout,
        de_h_view:        &wgpu::TextureView,
        accum_buf:        &wgpu::Buffer,
        params_buf:       &wgpu::Buffer,
        gradient_a_view:  &wgpu::TextureView,
        gradient_b_view:  &wgpu::TextureView,
        gradient_sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_light_bg"), layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(de_h_view),
                },
                wgpu::BindGroupEntry { binding: 1, resource: accum_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding:  3,
                    resource: wgpu::BindingResource::TextureView(gradient_a_view),
                },
                wgpu::BindGroupEntry {
                    binding:  4,
                    resource: wgpu::BindingResource::TextureView(gradient_b_view),
                },
                wgpu::BindGroupEntry {
                    binding:  5,
                    resource: wgpu::BindingResource::Sampler(gradient_sampler),
                },
            ],
        })
    }

    fn make_blit_bg(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout,
        hdr_view: &wgpu::TextureView, sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"), layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    // ---- texture helpers ----

    fn make_de_h_texture(device: &wgpu::Device, width: u32, height: u32)
        -> (wgpu::Texture, wgpu::TextureView)
    {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("de_h_texture"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          DE_H_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                           | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    fn make_hdr_texture(device: &wgpu::Device, width: u32, height: u32)
        -> (wgpu::Texture, wgpu::TextureView)
    {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("hdr_texture"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          HDR_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
                           | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    fn make_gradient_texture(device: &wgpu::Device, label: &str)
        -> (wgpu::Texture, wgpu::TextureView)
    {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some(label),
            size:            wgpu::Extent3d { width: GRADIENT_TEX_WIDTH, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    }

    // ---- public API ----

    /// Reinitialise all trajectory states with fresh ICs spread across the attractor.
    /// Call this when attractor type or parameters change.
    pub fn reset_sim_states(&self, queue: &wgpu::Queue, config: &AttractorConfig) {
        let states = make_attractor_states(SIM_NUM_TRAJECTORIES, config);
        self.attractor_seed.set(seed_from_states(&states));
        queue.write_buffer(&self.sim_state_buf, 0, bytemuck::cast_slice(&states));
    }

    /// Dispatch one GPU simulation step: advances all trajectories by
    /// `SIM_STEPS_PER_DISPATCH` Euler steps and splats them into the histogram.
    pub fn dispatch_sim(
        &self,
        queue:     &wgpu::Queue,
        encoder:   &mut wgpu::CommandEncoder,
        view_proj: Mat4,
        config:    &AttractorConfig,
    ) {
        let ss_w = self.width  * self.ss_scale;
        let ss_h = self.height * self.ss_scale;

        // Copy attractor params into the fixed-size array; unused slots stay 0.
        // Seed hint (p[44-46]) is only written when params don't extend into that
        // region — PolySprott has 169 params and manages its own IC seeding on GPU.
        let mut p = [0.0f32; 172];
        let n = config.params.len().min(172);
        p[..n].copy_from_slice(&config.params[..n]);
        if n <= 44 {
            let seed = self.attractor_seed.get();
            p[44] = seed[0];
            p[45] = seed[1];
            p[46] = seed[2];
        }

        queue.write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(&SimParams {
            view_proj: view_proj.to_cols_array(),
            ss_width:  ss_w,
            ss_height: ss_h,
            steps:     SIM_STEPS_PER_DISPATCH,
            num_traj:  SIM_NUM_TRAJECTORIES,
            p,
        }));

        let pipeline = &self.sim_pipelines[&config.attractor_type];
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim_pass"), timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &self.cached_sim_bg, &[]);
        let workgroups = (SIM_NUM_TRAJECTORIES + 255) / 256;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn clear(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.accum_buf, 0, None);
        encoder.clear_buffer(&self.max_density_buf, 0, None);
        self.last_max_density = 1;
        // Do NOT touch max_map_inflight / max_map_ready here.
    }

    /// Two-pass composite: horizontal DE blur then vertical DE blur + colorize.
    /// `render_mode` selects between monochrome and light-mode gradient compositing.
    pub fn composite(
        &self,
        queue:       &wgpu::Queue,
        encoder:     &mut wgpu::CommandEncoder,
        params:      CompositeParams,
        render_mode: RenderMode,
    ) {
        queue.write_buffer(&self.composite_params_buf, 0, bytemuck::bytes_of(&params));

        // Pass 1: horizontal DE blur → de_h_texture (R32Float).  Same for all modes.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("de_h_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &self.de_h_texture_view,
                    resolve_target: None,
                    ops:            wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });
            rpass.set_pipeline(&self.de_h_pipeline);
            rpass.set_bind_group(0, &self.cached_de_h_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Pass 2: vertical DE blur + colorize → hdr_texture (Rgba16Float).
        let (pipeline, bind_group) = match render_mode {
            RenderMode::Monochrome => (&self.composite_pipeline,       &self.cached_composite_bg),
            RenderMode::Light      => (&self.composite_light_pipeline, &self.cached_composite_light_bg),
        };
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &self.hdr_texture_view,
                    resolve_target: None,
                    ops:            wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }

    pub fn blit(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           target,
                resolve_target: None,
                ops:            wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
        });
        rpass.set_pipeline(&self.blit_pipeline);
        rpass.set_bind_group(0, &self.cached_blit_bg, &[]);
        rpass.draw(0..3, 0..1);
    }

    pub fn encode_max_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.max_map_inflight { return; }
        encoder.copy_buffer_to_buffer(&self.max_density_buf, 0, &self.max_readback_buf, 0, 4);
    }

    pub fn submit_max_readback(&mut self) {
        if self.max_map_inflight { return; }
        self.max_map_inflight = true;
        let flag = Arc::clone(&self.max_map_ready);
        self.max_readback_buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() { flag.store(true, Ordering::Release); }
        });
    }

    pub fn poll_max_density(&mut self, device: &wgpu::Device) {
        if !self.max_map_inflight { return; }
        device.poll(wgpu::Maintain::Poll);
        if self.max_map_ready.load(Ordering::Acquire) {
            let range = self.max_readback_buf.slice(..).get_mapped_range();
            let val = u32::from_le_bytes([range[0], range[1], range[2], range[3]]);
            drop(range);
            self.max_readback_buf.unmap();
            self.max_map_ready.store(false, Ordering::Release);
            self.max_map_inflight = false;
            if val > 0 { self.last_max_density = val; }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU compute pipeline helper
// ---------------------------------------------------------------------------

fn make_sim_pipeline(
    device:      &wgpu::Device,
    shader_src:  &str,
    bind_layout: &wgpu::BindGroupLayout,
    label:       &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some(label),
        bind_group_layouts:   &[bind_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:  Some(label),
        layout: Some(&pl),
        module: &shader,
        entry_point:         "main",
        compilation_options: Default::default(),
        cache:               None,
    })
}

// ---------------------------------------------------------------------------
// CPU-side attractor state initialisation
// ---------------------------------------------------------------------------

fn make_attractor_states(num: u32, config: &AttractorConfig) -> Vec<[f32; 4]> {
    match config.attractor_type {
        AttractorType::Lorenz      => make_lorenz_states(num, &config.params),
        AttractorType::Lorenz84    => make_lorenz84_states(num, &config.params),
        AttractorType::Rossler     => make_rossler_states(num, &config.params),
        AttractorType::Thomas      => make_thomas_states(num, &config.params),
        AttractorType::ChaoticFlow => make_chaotic_flow_states(num, &config.params),
        AttractorType::Pickover    => make_map_states(num, &config.params, |p, x, y, z| {
            let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
            ((a*y).sin() - z*(b*x).cos(), z*(c*x).sin() - (d*y).cos(), x.sin())
        }),
        AttractorType::Clifford    => make_map_states(num, &config.params, |p, x, y, _z| {
            let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
            ((a*y).sin() + c*(a*x).cos(), (b*x).sin() + d*(b*y).cos(), 0.0)
        }),
        AttractorType::Icon  => make_map_states_icon(num, &config.params),
        AttractorType::IconB => make_map_states_icon_b(num, &config.params),
        AttractorType::PolyA => make_map_states(num, &config.params, |p, x, y, z| {
            let (p0, p1, p2) = (p[0], p[1], p[2]);
            (p0 + y - z * y, p1 + z - x * z, p2 + x - y * x)
        }),
        AttractorType::PolyAbs => make_map_states(num, &config.params, |p, x, y, z| (
            p[0]  + p[1]*x  + p[2]*y  + p[3]*z  + p[4]*x.abs()  + p[5]*y.abs()  + p[6]*z.abs(),
            p[7]  + p[8]*x  + p[9]*y  + p[10]*z + p[11]*x.abs() + p[12]*y.abs() + p[13]*z.abs(),
            p[14] + p[15]*x + p[16]*y + p[17]*z + p[18]*x.abs() + p[19]*y.abs() + p[20]*z.abs(),
        )),
        AttractorType::PolyB => make_map_states(num, &config.params, |p, x, y, z| {
            (p[0] + y - z * (p[1] + y), p[2] + z - x * (p[3] + z), p[4] + x - y * (p[5] + x))
        }),
        AttractorType::PolyC => make_map_states(num, &config.params, |p, x, y, z| {
            (
                p[0] + x * (p[1] + p[2] * x + p[3] * y) + y * (p[4] + p[5] * y),
                p[6] + y * (p[7] + p[8] * y + p[9] * z) + z * (p[10] + p[11] * z),
                p[12] + z * (p[13] + p[14] * z + p[15] * x) + x * (p[16] + p[17] * x),
            )
        }),
        AttractorType::PolyPow => make_map_states(num, &config.params, |p, x, y, z| (
            p[0]  + p[1]*x  + p[2]*y  + p[3]*z  + p[4]*x.abs()  + p[5]*y.abs()  + p[6]*z.abs().powf(p[7]),
            p[8]  + p[9]*x  + p[10]*y + p[11]*z + p[12]*x.abs() + p[13]*y.abs() + p[14]*z.abs().powf(p[15]),
            p[16] + p[17]*x + p[18]*y + p[19]*z + p[20]*x.abs() + p[21]*y.abs() + p[22]*z.abs().powf(p[23]),
        )),
        AttractorType::PolySin => make_map_states(num, &config.params, |p, x, y, z| (
            p[0]  + p[1]*x  + p[2]*y  + p[3]*z
                  + p[4]*(p[5]*x+p[6]).sin()   + p[7]*(p[8]*y+p[9]).sin()   + p[10]*(p[11]*z+p[12]).sin(),
            p[13] + p[14]*x + p[15]*y + p[16]*z
                  + p[17]*(p[18]*x+p[19]).sin() + p[20]*(p[21]*y+p[22]).sin() + p[23]*(p[24]*z+p[25]).sin(),
            p[26] + p[27]*x + p[28]*y + p[29]*z
                  + p[30]*(p[31]*x+p[32]).sin() + p[33]*(p[34]*y+p[35]).sin() + p[36]*(p[37]*z+p[38]).sin(),
        )),
        AttractorType::PolySprott => make_map_states(num, &config.params, poly_sprott_eval),
    }
}

fn poly_sprott_eval(p: &[f32], x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let order = if p.is_empty() { 2 } else { p[0].round() as usize };
    let cx = if p.len() >= 57  { &p[1..57]   } else { &[] as &[f32] };
    let cy = if p.len() >= 113 { &p[57..113]  } else { &[] as &[f32] };
    let cz = if p.len() >= 169 { &p[113..169] } else { &[] as &[f32] };
    (poly3d(cx, x, y, z, order), poly3d(cy, x, y, z, order), poly3d(cz, x, y, z, order))
}

fn poly3d(c: &[f32], x: f32, y: f32, z: f32, order: usize) -> f32 {
    if c.len() < 10 { return 0.0; }
    let x2 = x*x; let y2 = y*y; let z2 = z*z;
    // Sprott ordering: 1, x, x², xy, xz, y, y², yz, z, z²
    let mut r = c[0] + c[1]*x + c[2]*x2 + c[3]*(x*y) + c[4]*(x*z)
              + c[5]*y + c[6]*y2 + c[7]*(y*z)
              + c[8]*z + c[9]*z2;
    if order >= 3 && c.len() >= 20 {
        let x3 = x2*x; let y3 = y2*y; let z3 = z2*z;
        r += c[10]*x3  + c[11]*(x2*y) + c[12]*(x2*z) + c[13]*(x*y2)
           + c[14]*(x*y*z) + c[15]*(x*z2) + c[16]*y3 + c[17]*(y2*z)
           + c[18]*(y*z2) + c[19]*z3;
        if order >= 4 && c.len() >= 35 {
            let x4 = x3*x; let y4 = y3*y; let z4 = z3*z;
            r += c[20]*x4  + c[21]*(x3*y) + c[22]*(x3*z) + c[23]*(x2*y2)
               + c[24]*(x2*y*z) + c[25]*(x2*z2) + c[26]*(x*y3) + c[27]*(x*y2*z)
               + c[28]*(x*y*z2) + c[29]*(x*z3) + c[30]*y4 + c[31]*(y3*z)
               + c[32]*(y2*z2) + c[33]*(y*z3) + c[34]*z4;
            if order >= 5 && c.len() >= 56 {
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

/// Generate `num` initial Lorenz trajectory states spread across the attractor.
/// Uses 32 warm-up groups with staggered ICs; adjacent samples 100 steps apart.
fn make_lorenz_states(num: u32, params: &[f32]) -> Vec<[f32; 4]> {
    let a  = params.get(0).copied().unwrap_or(16.227);
    let b  = params.get(1).copied().unwrap_or(15.223);
    let c  = params.get(2).copied().unwrap_or(8.018);
    let dt = params.get(3).copied().unwrap_or(0.049);

    let num_groups: u32 = 32;
    let per_group = (num + num_groups - 1) / num_groups;
    const INTER: u32 = 100;
    let mut states = Vec::with_capacity(num as usize);

    for g in 0..num_groups {
        let (mut x, mut y, mut z) = (0.1 + g as f32 * 0.7, g as f32 * 1.3, g as f32 * 2.1);
        for _ in 0..1_000 {
            let (nx, ny, nz) = (
                x + a * dt * (y - x),
                y + dt * (b * x - y - z * x),
                z + dt * (x * y - c * z),
            );
            (x, y, z) = (nx, ny, nz);
        }
        for _ in 0..per_group {
            if states.len() >= num as usize { break; }
            states.push([x, y, z, 0.0]);
            for _ in 0..INTER {
                let (nx, ny, nz) = (
                    x + a * dt * (y - x),
                    y + dt * (b * x - y - z * x),
                    z + dt * (x * y - c * z),
                );
                (x, y, z) = (nx, ny, nz);
            }
        }
    }
    states
}

fn make_lorenz84_states(num: u32, params: &[f32]) -> Vec<[f32; 4]> {
    let a  = params.get(0).copied().unwrap_or(0.25);
    let b  = params.get(1).copied().unwrap_or(4.0);
    let f  = params.get(2).copied().unwrap_or(8.0);
    let g  = params.get(3).copied().unwrap_or(1.0);
    let dt = params.get(4).copied().unwrap_or(0.01);

    let num_groups: u32 = 32;
    let per_group = (num + num_groups - 1) / num_groups;
    const INTER: u32 = 100;
    let mut states = Vec::with_capacity(num as usize);

    // Small jitter around a point inside the basin of attraction (~[-3,3]).
    // 2000 steps × dt≈0.01 ≈ 20 time units — enough to land on the attractor.
    for grp in 0..num_groups {
        let (mut x, mut y, mut z) = (
            0.1 + grp as f32 * 0.05,
            grp as f32 * 0.1,
            grp as f32 * 0.05,
        );
        for _ in 0..2000 {
            let (nx, ny, nz) = (
                x + dt * (-a * x - y * y - z * z + a * f),
                y + dt * (-y + x * y - b * x * z + g),
                z + dt * (-z + b * x * y + x * z),
            );
            if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                (x, y, z) = (nx, ny, nz);
            }
        }
        for _ in 0..per_group {
            if states.len() >= num as usize { break; }
            states.push([x, y, z, 0.0]);
            for _ in 0..INTER {
                let (nx, ny, nz) = (
                    x + dt * (-a * x - y * y - z * z + a * f),
                    y + dt * (-y + x * y - b * x * z + g),
                    z + dt * (-z + b * x * y + x * z),
                );
                if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                    (x, y, z) = (nx, ny, nz);
                }
            }
        }
    }
    states
}

fn make_rossler_states(num: u32, params: &[f32]) -> Vec<[f32; 4]> {
    let a  = params.get(0).copied().unwrap_or(0.2);
    let b  = params.get(1).copied().unwrap_or(0.2);
    let c  = params.get(2).copied().unwrap_or(5.7);
    let dt = params.get(3).copied().unwrap_or(0.05);

    let num_groups: u32 = 32;
    let per_group = (num + num_groups - 1) / num_groups;
    const INTER: u32 = 100;
    let mut states = Vec::with_capacity(num as usize);

    // Keep starting points within the attractor basin (x∈[-9,11], y∈[-10,5]).
    // Offsets grp*0.25 → max x=7.75; grp*0.1 → max y=3.1.
    // 1000 steps × dt≈0.05 ≈ 50 time units ≈ 8 Rössler periods — enough to settle.
    for grp in 0..num_groups {
        let (mut x, mut y, mut z) = (grp as f32 * 0.25, grp as f32 * 0.1, 0.1);
        for _ in 0..1000 {
            let (nx, ny, nz) = (
                x + dt * (-y - z),
                y + dt * (x + a * y),
                z + dt * (b + z * (x - c)),
            );
            if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                (x, y, z) = (nx, ny, nz);
            }
        }
        for _ in 0..per_group {
            if states.len() >= num as usize { break; }
            states.push([x, y, z, 0.0]);
            for _ in 0..INTER {
                let (nx, ny, nz) = (
                    x + dt * (-y - z),
                    y + dt * (x + a * y),
                    z + dt * (b + z * (x - c)),
                );
                if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                    (x, y, z) = (nx, ny, nz);
                }
            }
        }
    }
    states
}

fn make_thomas_states(num: u32, params: &[f32]) -> Vec<[f32; 4]> {
    let b  = params.get(0).copied().unwrap_or(0.208);
    let dt = params.get(1).copied().unwrap_or(0.05);

    let num_groups: u32 = 32;
    let per_group = (num + num_groups - 1) / num_groups;
    const INTER: u32 = 100;
    let mut states = Vec::with_capacity(num as usize);

    for grp in 0..num_groups {
        let (mut x, mut y, mut z) = (
            0.1 + grp as f32 * 0.4,
            grp as f32 * 0.6,
            grp as f32 * 0.2,
        );
        for _ in 0..500 {
            let (nx, ny, nz) = (
                x + dt * (-b * x + y.sin()),
                y + dt * (-b * y + z.sin()),
                z + dt * (-b * z + x.sin()),
            );
            if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                (x, y, z) = (nx, ny, nz);
            }
        }
        for _ in 0..per_group {
            if states.len() >= num as usize { break; }
            states.push([x, y, z, 0.0]);
            for _ in 0..INTER {
                let (nx, ny, nz) = (
                    x + dt * (-b * x + y.sin()),
                    y + dt * (-b * y + z.sin()),
                    z + dt * (-b * z + x.sin()),
                );
                if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                    (x, y, z) = (nx, ny, nz);
                }
            }
        }
    }
    states
}

fn make_map_states_icon(num: u32, params: &[f32]) -> Vec<[f32; 4]> {
    let degree = params.get(0).copied().unwrap_or(3.0).max(1.0) as u32;
    let (alpha, beta, lambda, gamma, omega) = (
        params.get(1).copied().unwrap_or(-2.08),
        params.get(2).copied().unwrap_or(0.4),
        params.get(3).copied().unwrap_or(0.04),
        params.get(4).copied().unwrap_or(0.1),
        params.get(5).copied().unwrap_or(0.0),
    );
    let icon_step = |x: f32, y: f32| -> (f32, f32, f32) {
        let (mut re_r, mut im_r) = (1.0f32, 0.0f32);
        let (mut re_prev, mut im_prev) = (1.0f32, 0.0f32);
        for _ in 0..degree {
            re_prev = re_r; im_prev = im_r;
            let nr = re_r * x - im_r * y;
            let ni = re_r * y + im_r * x;
            re_r = nr; im_r = ni;
        }
        let r_sq = x * x + y * y;
        let p = lambda + alpha * r_sq + beta * re_r;
        (
            p * x + gamma * re_prev + omega * im_prev,
            p * y + omega * re_prev - gamma * im_prev,
            r_sq.sqrt(),
        )
    };

    // Standard Field-Golubitsky: equilibrium at R^2 = (1-lambda)/alpha
    let rhs = if alpha.abs() > 1e-6 { (1.0 - lambda) / alpha } else { 0.0 };
    let r_eq = if rhs > 0.0 { rhs.sqrt().clamp(0.01, 2.0) } else { 0.5_f32 };

    let num_groups: u32 = 32;
    let per_group = (num + num_groups - 1) / num_groups;
    const INTER: u32 = 50;
    const WARM: u32 = 500;
    let mut states = Vec::with_capacity(num as usize);

    for grp in 0..num_groups {
        let (mut x, mut y) = (r_eq + grp as f32 * 0.01, grp as f32 * 0.005);
        let mut z = 0.0f32;
        for _ in 0..WARM {
            let (nx, ny, nz) = icon_step(x, y);
            if nx.is_finite() && ny.is_finite() {
                (x, y, z) = (nx, ny, nz);
            } else {
                (x, y, z) = (r_eq, 0.0, 0.0);
            }
        }
        for _ in 0..per_group {
            if states.len() >= num as usize { break; }
            states.push([x, y, z, 0.0]);
            for _ in 0..INTER {
                let (nx, ny, nz) = icon_step(x, y);
                if nx.is_finite() && ny.is_finite() {
                    (x, y, z) = (nx, ny, nz);
                }
            }
        }
    }
    states
}

fn make_map_states_icon_b(num: u32, params: &[f32]) -> Vec<[f32; 4]> {
    let degree = params.get(0).copied().unwrap_or(2.0).max(1.0) as u32;
    let (alpha, beta, lambda, gamma, omega) = (
        params.get(1).copied().unwrap_or(-1.715),
        params.get(2).copied().unwrap_or(-0.689),
        params.get(3).copied().unwrap_or(2.343),
        params.get(4).copied().unwrap_or(0.351),
        params.get(5).copied().unwrap_or(-0.428),
    );
    let icon_b_step = |x: f32, y: f32| -> (f32, f32, f32) {
        let (mut re_r, mut im_r) = (1.0f32, 0.0f32);
        for _ in 0..degree {
            let nr = re_r * x - im_r * y;
            let ni = re_r * y + im_r * x;
            re_r = nr; im_r = ni;
        }
        let norm_v = (x * x + y * y).sqrt();
        let p = lambda + alpha * norm_v + beta * (x * re_r - y * im_r);
        (
            p * x + gamma * re_r - omega * y,
            p * y - gamma * im_r + omega * x,
            norm_v,
        )
    };

    // Equilibrium radius: λ + α·R = 1  →  R = (1-λ)/α
    let rhs = if alpha.abs() > 1e-6 { (1.0 - lambda) / alpha } else { 0.0 };
    let r_eq = if rhs > 0.0 { rhs.clamp(0.01, 2.0) } else { 0.5_f32 };

    let num_groups: u32 = 32;
    let per_group = (num + num_groups - 1) / num_groups;
    const INTER: u32 = 50;
    const WARM: u32 = 500;
    let mut states = Vec::with_capacity(num as usize);

    for grp in 0..num_groups {
        let (mut x, mut y) = (r_eq + grp as f32 * 0.01, grp as f32 * 0.005);
        let mut z = 0.0f32;
        for _ in 0..WARM {
            let (nx, ny, nz) = icon_b_step(x, y);
            if nx.is_finite() && ny.is_finite() {
                (x, y, z) = (nx, ny, nz);
            } else {
                (x, y, z) = (r_eq, 0.0, 0.0);
            }
        }
        for _ in 0..per_group {
            if states.len() >= num as usize { break; }
            states.push([x, y, z, 0.0]);
            for _ in 0..INTER {
                let (nx, ny, nz) = icon_b_step(x, y);
                if nx.is_finite() && ny.is_finite() {
                    (x, y, z) = (nx, ny, nz);
                }
            }
        }
    }
    states
}

/// Extract a seed (a known-good point on the attractor) from a state list.
/// Uses the midpoint of the list so it is well past any transient.
fn seed_from_states(states: &[[f32; 4]]) -> [f32; 3] {
    let s = states.get(states.len() / 2).copied().unwrap_or([0.0, 0.0, 0.0, 0.0]);
    [s[0], s[1], s[2]]
}

/// Probe a set of candidate initial conditions for `f`, running up to `WARM` steps
/// each, and return the endpoint of whichever candidate survives the longest.
/// This handles attractors whose basin of attraction does not include the origin.
fn probe_map_seed<F>(f: &F, params: &[f32]) -> (f32, f32, f32)
where
    F: Fn(&[f32], f32, f32, f32) -> (f32, f32, f32),
{
    const WARM: u32 = 500;
    // Candidates span a range of z values and a few off-axis points so that
    // attractors elevated away from z=0 are reliably found.
    const CANDIDATES: &[(f32, f32, f32)] = &[
        (0.1, 0.1, 0.0), (0.1, 0.1, 0.5), (0.1, 0.1, 1.0),
        (0.1, 0.1, 1.5), (0.1, 0.1, 2.0), (0.1, 0.1, 2.5),
        (0.5, 0.5, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 1.0),
        (0.5, 0.5, 1.5), (0.5, 0.5, 2.0),
        (-0.3, 0.5, 0.5), (0.5, -0.3, 0.5), (0.5, 0.5, -0.5),
    ];
    let mut best = (0.1_f32, 0.1, 0.1);
    let mut best_count = 0u32;
    for &(cx, cy, cz) in CANDIDATES {
        let (mut x, mut y, mut z) = (cx, cy, cz);
        let mut count = 0u32;
        for _ in 0..WARM {
            let (nx, ny, nz) = f(params, x, y, z);
            if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                (x, y, z) = (nx, ny, nz);
                count += 1;
            } else {
                break;
            }
        }
        if count > best_count {
            best_count = count;
            best = (x, y, z);
        }
        if count >= WARM {
            break; // Found a fully-bounded IC; no need to try more
        }
    }
    best
}

/// Generic state generator for discrete maps.
/// Probes multiple initial conditions to find one inside the basin of attraction,
/// then generates `num` states spaced 50 steps apart along the attractor.
fn make_map_states<F>(num: u32, params: &[f32], f: F) -> Vec<[f32; 4]>
where
    F: Fn(&[f32], f32, f32, f32) -> (f32, f32, f32),
{
    const INTER: u32 = 50;
    let (mut x, mut y, mut z) = probe_map_seed(&f, params);
    let mut states = Vec::with_capacity(num as usize);
    while states.len() < num as usize {
        states.push([x, y, z, 0.0]);
        for _ in 0..INTER {
            let (nx, ny, nz) = f(params, x, y, z);
            if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                (x, y, z) = (nx, ny, nz);
            }
        }
    }
    states
}

fn make_chaotic_flow_states(num: u32, params: &[f32]) -> Vec<[f32; 4]> {
    // Layout: [0..11]=m0..m11, [12..14]=Op0..Op2, [15..17]=Op4..Op6,
    //         [18..20]=Op8..Op10, [21]=dt  (m3, m7, m11 are always constant)
    let get = |i: usize| params.get(i).copied().unwrap_or(0.0);
    let m   = |i: usize| get(i);
    let op  = |i: usize| get(12 + i) as u32;  // i=0..8 maps to Op0,Op1,Op2,Op4,Op5,Op6,Op8,Op9,Op10
    let dt  = get(21).max(1e-6_f32);

    let eval_v = |o: u32, x: f32, y: f32, z: f32| -> f32 {
        match o { 1 => x, 2 => y, 3 => z, _ => 1.0 }
    };

    let step = |x: f32, y: f32, z: f32| -> (f32, f32, f32) {
        let dx = m(0)*eval_v(op(0),x,y,z)*x + m(1)*eval_v(op(1),x,y,z)*y
               + m(2)*eval_v(op(2),x,y,z)*z + m(3);
        let dy = m(4)*eval_v(op(3),x,y,z)*x + m(5)*eval_v(op(4),x,y,z)*y
               + m(6)*eval_v(op(5),x,y,z)*z + m(7);
        let dz = m(8)*eval_v(op(6),x,y,z)*x + m(9)*eval_v(op(7),x,y,z)*y
               + m(10)*eval_v(op(8),x,y,z)*z + m(11);
        (x + dt*dx, y + dt*dy, z + dt*dz)
    };

    let num_groups: u32 = 32;
    let per_group = (num + num_groups - 1) / num_groups;
    const INTER: u32 = 50;
    let mut states = Vec::with_capacity(num as usize);

    for grp in 0..num_groups {
        // Start near a mid-range point; 1000 steps at the param dt lands on the attractor.
        let (mut x, mut y, mut z) = (
            grp as f32 * 0.05,
            0.1 + grp as f32 * 0.05,
            grp as f32 * 0.03,
        );
        for _ in 0..1000 {
            let (nx, ny, nz) = step(x, y, z);
            if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                (x, y, z) = (nx, ny, nz);
            } else {
                (x, y, z) = (0.0, 0.1 + grp as f32 * 0.05, 0.0);
            }
        }
        for _ in 0..per_group {
            if states.len() >= num as usize { break; }
            states.push([x, y, z, 0.0]);
            for _ in 0..INTER {
                let (nx, ny, nz) = step(x, y, z);
                if nx.is_finite() && ny.is_finite() && nz.is_finite() {
                    (x, y, z) = (nx, ny, nz);
                }
            }
        }
    }
    states
}


// ---------------------------------------------------------------------------
// Bind group layout entry helpers
// ---------------------------------------------------------------------------

fn bgl_storage_r(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}

fn bgl_uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
        },
        count: None,
    }
}
