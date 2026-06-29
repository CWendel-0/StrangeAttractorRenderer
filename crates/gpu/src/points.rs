//! Points render mode: screen-space "fake 3D" depth/hit accumulation,
//! bounded by canvas resolution (not point count) so it can accumulate
//! indefinitely as cheaply as the density histogram does. Three passes per
//! frame: a bilateral gap-fill compute pass (`points_fill.wgsl`) over
//! `Histogram`'s raw camera-space depth/hit buffers; a shading pass
//! (`points_shade.wgsl`) that reconstructs and shades a fake surface at
//! full *supersample* resolution, one independent shading evaluation per
//! texel (so two overlapping strands never have their geometry blended
//! together, only their colors -- this is what keeps self-overlap
//! boundaries smooth); then a downsample composite pass
//! (`points_composite.wgsl`) that blends the shaded buffer down to canvas
//! resolution, writing directly into `canvas_texture` like Solid mode does.

const CANVAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SHADED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const FILL_WORKGROUP: u32 = 8;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointsParams {
    pub view_proj:          [f32; 16],
    pub inverse_view_proj:  [f32; 16],
    pub light_view_proj:    [f32; 16], // orthographic, framing the scene from the light's side
    pub camera_pos:         [f32; 4],  // xyz = world-space eye position
    pub light_dir_ambient:  [f32; 4],  // xyz = unit vector toward the light, w = ambient term
    pub base_color_alpha:   [f32; 4],  // xyz = material color, w = opacity
    pub specular_shininess: [f32; 4],  // xyz = specular/rim color, w = max shininess exponent (at roughness=0)
    pub material_extra:     [f32; 4],  // x = roughness [0,1], y = metalness [0,1], z = shading model id, w = anisotropy [-1,1]
    pub reflect_refract:    [f32; 4],  // x = reflectivity / dielectric F0, y = IOR, z = refraction strength, w unused
    pub sky_top:            [f32; 4],  // xyz = sky color looking "up", w unused
    pub sky_bottom:         [f32; 4],  // xyz = sky color looking "down", w unused
    pub model_params:       [f32; 4],  // meaning depends on shading model -- see shading_common.wgsl
    pub canvas_a:           [u32; 4],  // x = canvas width, y = canvas height, z = ss_scale, w = ss_width
    pub canvas_b:           [u32; 4],  // x = ss_height, y = light_buf_size, zw unused
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FillParams {
    ss_width:  u32,
    ss_height: u32,
    _pad:      [u32; 2],
}

/// Mirrors `points_composite.wgsl`'s `CompositeParams` -- just the canvas/ss
/// dimensions the downsample pass needs, a subset of the full `PointsParams`
/// the shade pass uses.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    canvas_a: [u32; 4],
    canvas_b: [u32; 4],
}

pub struct PointsRenderer {
    fill_pipeline:    wgpu::ComputePipeline,
    fill_bind_layout: wgpu::BindGroupLayout,
    fill_bind_group:  wgpu::BindGroup,
    fill_params_buf:  wgpu::Buffer,

    shade_pipeline:    wgpu::RenderPipeline,
    shade_bind_layout: wgpu::BindGroupLayout,
    shade_bind_group:  wgpu::BindGroup,
    shade_params_buf:  wgpu::Buffer,

    composite_pipeline:    wgpu::RenderPipeline,
    composite_bind_layout: wgpu::BindGroupLayout,
    composite_bind_group:  wgpu::BindGroup,
    composite_params_buf:  wgpu::Buffer,

    depth_filled_buf: wgpu::Buffer,
    hit_filled_buf:   wgpu::Buffer,

    shaded_tex:      wgpu::Texture,
    shaded_tex_view: wgpu::TextureView,

    ss_width:  u32,
    ss_height: u32,
}

impl PointsRenderer {
    pub fn new(
        device: &wgpu::Device,
        width: u32, height: u32, ss_scale: u32,
        points_depth_buf: &wgpu::Buffer, points_hit_buf: &wgpu::Buffer, points_light_depth_buf: &wgpu::Buffer,
    ) -> Self {
        let ss_width  = width  * ss_scale;
        let ss_height = height * ss_scale;

        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("points_fill_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/points_fill.wgsl").into()),
        });

        let fill_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("points_fill_params_buf"),
            size:               std::mem::size_of::<FillParams>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fill_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("points_fill_bgl"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::COMPUTE),
                bgl_storage_r(1, wgpu::ShaderStages::COMPUTE),
                bgl_storage_r(2, wgpu::ShaderStages::COMPUTE),
                bgl_storage_rw(3, wgpu::ShaderStages::COMPUTE),
                bgl_storage_rw(4, wgpu::ShaderStages::COMPUTE),
            ],
        });

        let fill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:  Some("points_fill_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("points_fill_pl"),
                bind_group_layouts:   &[&fill_bind_layout],
                push_constant_ranges: &[],
            })),
            module:              &fill_shader,
            entry_point:         "fill_main",
            compilation_options: Default::default(),
            cache:               None,
        });

        let (depth_filled_buf, hit_filled_buf) = Self::make_filled_bufs(device, ss_width, ss_height);

        let fill_bind_group = Self::make_fill_bind_group(
            device, &fill_bind_layout, &fill_params_buf,
            points_depth_buf, points_hit_buf, &depth_filled_buf, &hit_filled_buf,
        );

        // shading_common.wgsl holds the BRDF/shadow-math-adjacent code shared
        // with Solid mode's tube-mesh renderer (solid.rs); WGSL has no native
        // #include, so the two source files are concatenated here instead.
        let shade_source = format!(
            "{}\n{}",
            include_str!("../shaders/shading_common.wgsl"),
            include_str!("../shaders/points_shade.wgsl"),
        );
        let shade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("points_shade_shader"),
            source: wgpu::ShaderSource::Wgsl(shade_source.into()),
        });

        let shade_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("points_shade_params_buf"),
            size:               std::mem::size_of::<PointsParams>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shade_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("points_shade_bgl"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::FRAGMENT),
                bgl_storage_r(1, wgpu::ShaderStages::FRAGMENT),
                bgl_storage_r(2, wgpu::ShaderStages::FRAGMENT),
                bgl_storage_r(3, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let shade_bind_group = Self::make_shade_bind_group(
            device, &shade_bind_layout, &shade_params_buf,
            &depth_filled_buf, &hit_filled_buf, points_light_depth_buf,
        );

        let shade_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("points_shade_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("points_shade_pl"),
                bind_group_layouts:   &[&shade_bind_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module:              &shade_shader,
                entry_point:         "vs_main",
                buffers:             &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shade_shader,
                entry_point: "fs_main",
                targets:     &[Some(wgpu::ColorTargetState {
                    format:     SHADED_FORMAT,
                    blend:      None, // one invocation per texel, no overlapping geometry to blend
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:     wgpu::PrimitiveState::default(),
            depth_stencil: None, // single fullscreen pass, no overlapping primitives to depth-test
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        let (shaded_tex, shaded_tex_view) = Self::make_shaded_tex(device, ss_width, ss_height);

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("points_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/points_composite.wgsl").into()),
        });

        let composite_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("points_composite_params_buf"),
            size:               std::mem::size_of::<CompositeParams>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let composite_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("points_composite_bgl"),
            entries: &[
                bgl_uniform(0, wgpu::ShaderStages::FRAGMENT),
                bgl_texture(1, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let composite_bind_group = Self::make_composite_bind_group(
            device, &composite_bind_layout, &composite_params_buf, &shaded_tex_view,
        );

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("points_composite_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("points_composite_pl"),
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
                    format:     CANVAS_FORMAT,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:     wgpu::PrimitiveState::default(),
            depth_stencil: None, // single fullscreen pass, no overlapping primitives to depth-test
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        Self {
            fill_pipeline,
            fill_bind_layout,
            fill_bind_group,
            fill_params_buf,
            shade_pipeline,
            shade_bind_layout,
            shade_bind_group,
            shade_params_buf,
            composite_pipeline,
            composite_bind_layout,
            composite_bind_group,
            composite_params_buf,
            depth_filled_buf,
            hit_filled_buf,
            shaded_tex,
            shaded_tex_view,
            ss_width,
            ss_height,
        }
    }

    fn make_filled_bufs(device: &wgpu::Device, ss_width: u32, ss_height: u32) -> (wgpu::Buffer, wgpu::Buffer) {
        let make = |label| device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(label),
            size:               ss_width as u64 * ss_height as u64 * 4, // one u32 per texel
            usage:              wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        (make("points_depth_filled_buf"), make("points_hit_filled_buf"))
    }

    fn make_fill_bind_group(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout, params_buf: &wgpu::Buffer,
        raw_depth: &wgpu::Buffer, raw_hit: &wgpu::Buffer, filled_depth: &wgpu::Buffer, filled_hit: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("points_fill_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: raw_depth.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: raw_hit.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: filled_depth.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: filled_hit.as_entire_binding() },
            ],
        })
    }

    fn make_shade_bind_group(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout, params_buf: &wgpu::Buffer,
        filled_depth: &wgpu::Buffer, filled_hit: &wgpu::Buffer, light_depth: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("points_shade_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: filled_depth.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: filled_hit.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: light_depth.as_entire_binding() },
            ],
        })
    }

    fn make_shaded_tex(device: &wgpu::Device, ss_width: u32, ss_height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("points_shaded_tex"),
            size:            wgpu::Extent3d { width: ss_width.max(1), height: ss_height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          SHADED_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn make_composite_bind_group(
        device: &wgpu::Device, layout: &wgpu::BindGroupLayout, params_buf: &wgpu::Buffer,
        shaded_tex_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("points_composite_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(shaded_tex_view) },
            ],
        })
    }

    /// Recreate the filled depth/hit buffers and the shaded texture, and
    /// re-bind all three passes against `Histogram`'s current raw buffers.
    /// Call on canvas resize *and* on ss_scale change (both change
    /// `ss_width`/`ss_height`, and `Histogram` recreates its own raw buffers
    /// in both cases too).
    pub fn resize(
        &mut self, device: &wgpu::Device,
        width: u32, height: u32, ss_scale: u32,
        points_depth_buf: &wgpu::Buffer, points_hit_buf: &wgpu::Buffer, points_light_depth_buf: &wgpu::Buffer,
    ) {
        self.ss_width  = width  * ss_scale;
        self.ss_height = height * ss_scale;
        let (depth_filled_buf, hit_filled_buf) = Self::make_filled_bufs(device, self.ss_width, self.ss_height);
        self.depth_filled_buf = depth_filled_buf;
        self.hit_filled_buf   = hit_filled_buf;
        let (shaded_tex, shaded_tex_view) = Self::make_shaded_tex(device, self.ss_width, self.ss_height);
        self.shaded_tex      = shaded_tex;
        self.shaded_tex_view = shaded_tex_view;

        self.fill_bind_group = Self::make_fill_bind_group(
            device, &self.fill_bind_layout, &self.fill_params_buf,
            points_depth_buf, points_hit_buf, &self.depth_filled_buf, &self.hit_filled_buf,
        );
        self.shade_bind_group = Self::make_shade_bind_group(
            device, &self.shade_bind_layout, &self.shade_params_buf,
            &self.depth_filled_buf, &self.hit_filled_buf, points_light_depth_buf,
        );
        self.composite_bind_group = Self::make_composite_bind_group(
            device, &self.composite_bind_layout, &self.composite_params_buf, &self.shaded_tex_view,
        );
    }

    /// Bilateral gap-fill (compute) + per-supersample-texel shading (render,
    /// into `shaded_tex`) + downsample composite (render, into the canvas) --
    /// the Points-mode replacement for dispatch_sim's histogram path's
    /// composite + blit_to_canvas.
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        canvas_view: &wgpu::TextureView,
        params: PointsParams,
        bg_color: wgpu::Color,
    ) {
        queue.write_buffer(&self.fill_params_buf, 0, bytemuck::bytes_of(&FillParams {
            ss_width:  self.ss_width,
            ss_height: self.ss_height,
            _pad:      [0; 2],
        }));
        queue.write_buffer(&self.shade_params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.composite_params_buf, 0, bytemuck::bytes_of(&CompositeParams {
            canvas_a: params.canvas_a,
            canvas_b: params.canvas_b,
        }));

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("points_fill_pass"), timestamp_writes: None,
            });
            cpass.set_pipeline(&self.fill_pipeline);
            cpass.set_bind_group(0, &self.fill_bind_group, &[]);
            let wx = (self.ss_width  + FILL_WORKGROUP - 1) / FILL_WORKGROUP;
            let wy = (self.ss_height + FILL_WORKGROUP - 1) / FILL_WORKGROUP;
            cpass.dispatch_workgroups(wx, wy, 1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("points_shade_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &self.shaded_tex_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });
            rpass.set_pipeline(&self.shade_pipeline);
            rpass.set_bind_group(0, &self.shade_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("points_composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           canvas_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(bg_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
        });
        rpass.set_pipeline(&self.composite_pipeline);
        rpass.set_bind_group(0, &self.composite_bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}

fn bgl_uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_r(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_texture(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    }
}
