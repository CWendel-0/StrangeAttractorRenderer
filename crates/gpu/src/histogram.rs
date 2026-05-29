use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use glam::Mat4;

/// Interleaved u32 pairs: [density, speed_fixed] per pixel.
/// density (index 0) accumulates bilinear weights; speed (index 1) accumulates
/// log-encoded speed × weight / SPEED_SCALE — see sim.wgsl for details.
const ACCUM_ELEM_SIZE: u64 = 8;

/// Default Lorenz parameters [σ, ρ, β, dt] — must match lorenz.rs::DESCS defaults.
const LORENZ_DEFAULTS: [f32; 4] = [16.227, 15.223, 8.018, 0.049];

#[inline]
fn lorenz_param(params: &[f32], i: usize) -> f32 {
    params.get(i).copied().unwrap_or(LORENZ_DEFAULTS[i])
}
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    view_proj: [f32; 16], // 64 bytes
    ss_width:  u32,
    ss_height: u32,
    steps:     u32,
    num_traj:  u32,
    a:         f32,
    b:         f32,
    c:         f32,
    dt:        f32,
    // total 96 bytes ✓
}

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

    // ---- GPU simulation (Lorenz on GPU) ----
    sim_pipeline:    wgpu::ComputePipeline,
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
}

impl Histogram {
    pub fn new(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        width:  u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        initial_lorenz_params: &[f32],
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

        // ---- GPU simulation pipeline ----
        let sim_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("sim_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sim.wgsl").into()),
        });

        let sim_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("sim_bgl"),
            entries: &[
                bgl_storage_rw(0, wgpu::ShaderStages::COMPUTE), // states
                bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE), // accum
                bgl_storage_rw(2, wgpu::ShaderStages::COMPUTE), // max_density
                bgl_uniform(3,    wgpu::ShaderStages::COMPUTE), // sim_params
            ],
        });

        let sim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:  Some("sim_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("sim_pl"),
                bind_group_layouts:   &[&sim_bind_layout],
                push_constant_ranges: &[],
            })),
            module:              &sim_shader,
            entry_point:         "main",
            compilation_options: Default::default(),
            cache:               None,
        });

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

        // Upload initial trajectory states (spread across the attractor).
        let initial_states = make_lorenz_states(SIM_NUM_TRAJECTORIES, initial_lorenz_params);
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
            sim_pipeline,
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
    /// Call this when the attractor parameters change.
    pub fn reset_sim_states(&self, queue: &wgpu::Queue, lorenz_params: &[f32]) {
        let states = make_lorenz_states(SIM_NUM_TRAJECTORIES, lorenz_params);
        queue.write_buffer(&self.sim_state_buf, 0, bytemuck::cast_slice(&states));
    }

    /// Dispatch one GPU simulation step: advances all trajectories by
    /// `SIM_STEPS_PER_DISPATCH` Euler steps and splats them into the histogram.
    pub fn dispatch_sim(
        &self,
        queue:         &wgpu::Queue,
        encoder:       &mut wgpu::CommandEncoder,
        view_proj:     Mat4,
        lorenz_params: &[f32],
    ) {
        let ss_w = self.width  * self.ss_scale;
        let ss_h = self.height * self.ss_scale;
        queue.write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(&SimParams {
            view_proj: view_proj.to_cols_array(),
            ss_width:  ss_w,
            ss_height: ss_h,
            steps:     SIM_STEPS_PER_DISPATCH,
            num_traj:  SIM_NUM_TRAJECTORIES,
            a:  lorenz_param(lorenz_params, 0),
            b:  lorenz_param(lorenz_params, 1),
            c:  lorenz_param(lorenz_params, 2),
            dt: lorenz_param(lorenz_params, 3),
        }));

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim_pass"), timestamp_writes: None,
        });
        cpass.set_pipeline(&self.sim_pipeline);
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
// CPU-side Lorenz state initialisation
// ---------------------------------------------------------------------------

/// Generate `num` initial trajectory states spread across the Lorenz attractor.
///
/// Samples from 32 independent warm-up trajectories (different ICs) so the initial
/// positions are distributed across distinct attractor regions rather than clustered
/// along a single arc.  Within each group, adjacent samples are 100 steps apart —
/// well beyond the Lyapunov time (~22 steps for default params) — so they diverge
/// immediately rather than staying correlated for the first few frames.
fn make_lorenz_states(num: u32, params: &[f32]) -> Vec<[f32; 4]> {
    let a  = lorenz_param(params, 0);
    let b  = lorenz_param(params, 1);
    let c  = lorenz_param(params, 2);
    let dt = lorenz_param(params, 3);

    let num_groups: u32 = 32;
    let per_group = (num + num_groups - 1) / num_groups;
    const INTER_SAMPLE_STEPS: u32 = 100;

    let mut states = Vec::with_capacity(num as usize);

    for g in 0..num_groups {
        // Stagger starting ICs so each group begins in a different attractor region.
        let (mut x, mut y, mut z) = (
            0.1 + g as f32 * 0.7,
            g as f32 * 1.3,
            g as f32 * 2.1,
        );

        // Warm up to settle onto the attractor.
        for _ in 0..1_000 {
            let nx = x + a * dt * (y - x);
            let ny = y + dt * (b * x - y - z * x);
            let nz = z + dt * (x * y - c * z);
            (x, y, z) = (nx, ny, nz);
        }

        for _ in 0..per_group {
            if states.len() >= num as usize { break; }
            states.push([x, y, z, 0.0]);
            for _ in 0..INTER_SAMPLE_STEPS {
                let nx = x + a * dt * (y - x);
                let ny = y + dt * (b * x - y - z * x);
                let nz = z + dt * (x * y - c * z);
                (x, y, z) = (nx, ny, nz);
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
