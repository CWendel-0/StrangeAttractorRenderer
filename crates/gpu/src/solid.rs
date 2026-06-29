//! Solid render mode: rasterizes a CPU-built tube mesh (see `sim::mesh`)
//! into the same `canvas_texture` Monochrome/Light blit into, but via a real
//! vertex+fragment pipeline with MSAA, depth testing, a shadow-mapped
//! directional light, and Blinn-Phong shading -- instead of the
//! dispatch_sim/composite/blit histogram path.

const DEPTH_FORMAT:   wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CANVAS_FORMAT:  wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const MSAA_SAMPLES:   u32 = 4;
/// Fixed shadow-map resolution, independent of canvas size -- must match the
/// `texel` constant in solid.wgsl's PCF loop.
const SHADOW_MAP_SIZE: u32 = 2048;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SolidParams {
    pub view_proj:          [f32; 16],
    pub light_view_proj:    [f32; 16], // orthographic, framing the mesh from the light's side
    pub camera_pos:         [f32; 4],  // xyz = world-space eye position
    pub light_dir_ambient:  [f32; 4],  // xyz = unit vector toward the light, w = ambient term
    pub base_color_alpha:   [f32; 4],  // xyz = material color, w = opacity
    pub specular_shininess: [f32; 4],  // xyz = specular color, w = max shininess exponent (at roughness=0)
    pub material_extra:     [f32; 4],  // x = roughness [0,1], y = metalness [0,1] (Cook-Torrance/Anisotropic GGX), z = shading model id, w = anisotropy [-1,1] (Anisotropic GGX)
    pub reflect_refract:    [f32; 4],  // x = reflectivity (Fresnel F0), y = IOR, z = refraction strength, w unused
    pub sky_top:            [f32; 4],  // xyz = sky color looking "up", w unused
    pub sky_bottom:         [f32; 4],  // xyz = sky color looking "down", w unused
    pub model_params:       [f32; 4],  // meaning depends on shading model -- see shading_common.wgsl
}

pub struct SolidRenderer {
    pipeline:        wgpu::RenderPipeline,
    bind_group:      wgpu::BindGroup,
    params_buf:      wgpu::Buffer,

    shadow_pipeline:    wgpu::RenderPipeline,
    shadow_bind_group:  wgpu::BindGroup,
    shadow_map_view:    wgpu::TextureView,

    msaa_color_view:    wgpu::TextureView,
    depth_texture_view: wgpu::TextureView,

    vertex_buf:      wgpu::Buffer,
    index_buf:       wgpu::Buffer,
    vertex_capacity: u64, // bytes currently allocated
    index_capacity:  u64, // bytes currently allocated
    index_count:     u32,
}

impl SolidRenderer {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // shading_common.wgsl holds the BRDF/shadow/sky code shared with
        // Points mode's sphere-impostor renderer (points.rs); WGSL has no
        // native #include, so the two source files are concatenated here at
        // shader-module-creation time instead.
        let source = format!(
            "{}\n{}",
            include_str!("../shaders/shading_common.wgsl"),
            include_str!("../shaders/solid.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("solid_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("solid_params_buf"),
            size:               std::mem::size_of::<SolidParams>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_map_view = Self::make_shadow_map_view(device);

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("solid_shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            compare:        Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Main pass: uniform params + shadow map texture/sampler.
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("solid_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty:         wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Texture {
                        multisampled:   false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type:    wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("solid_bg"),
            layout:  &bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_map_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });

        // Shadow pass: just the uniform params (for light_view_proj), no fragment stage.
        let shadow_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("solid_shadow_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty:         wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("solid_shadow_bg"),
            layout:  &shadow_bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() }],
        });

        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<sim::TubeVertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
            ],
        }];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("solid_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("solid_pl"),
                bind_group_layouts:   &[&bind_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_main",
                targets:     &[Some(wgpu::ColorTargetState {
                    format: CANVAS_FORMAT,
                    // Always blend (even at alpha=1, where it's a no-op
                    // equivalent to opaque drawing) so a single pipeline
                    // covers both the opaque and transparent material cases
                    // exposed in the UI. Overlapping tube strands aren't
                    // depth-sorted, so transparent materials may show
                    // order-dependent blending artifacts -- an accepted v1
                    // simplification.
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            // Cull back faces: the ring-sweep winding is consistently
            // outward-facing (verified analytically -- each ring quad's
            // cross product matches its vertices' outward radial normal),
            // so culling removes only each local cylinder's hidden interior
            // wall. Without this, transparent materials showed both the near
            // and far tube walls blended together, making every ring seam
            // visible as a banding artifact. Distinct strands of the curve
            // that happen to overlap in screen space are unaffected -- each
            // still draws its own near-facing surface.
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  Some(wgpu::Face::Back),
                ..Default::default()
            },
            // Depth writes always on, even for translucent material: with a
            // single depth buffer, only one surface per pixel can be shown
            // correctly. Disabling depth writes for alpha<1 was tried and
            // reverted -- it let overlapping strands blend together more
            // smoothly, but at the cost of resolving strictly in mesh-index
            // order instead of camera distance, so strands that should
            // clearly occlude others (closer to the camera) could be drawn
            // over by farther ones. Correct front-to-back ordering matters
            // more than smooth overlap blending; the latter would need real
            // order-independent transparency (e.g. weighted-blended OIT), not
            // a depth-state toggle, see git history for context.
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLES,
                ..Default::default()
            },
            multiview: None,
            cache:     None,
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("solid_shadow_pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("solid_shadow_pl"),
                bind_group_layouts:   &[&shadow_bind_layout],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_shadow",
                buffers:     &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: None,
            // No culling for the depth-only pass: the tube is thin relative
            // to typical light-to-mesh distances, and culling back faces
            // here (relative to the *light's* view) risks light leaking
            // through near-edge-on geometry (classic shadow "peter-panning").
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(), // no MSAA for the shadow map
            multiview:   None,
            cache:       None,
        });

        let msaa_color_view = Self::make_msaa_color_view(device, width, height);
        let depth_texture_view = Self::make_depth_view(device, width, height);

        // Empty placeholder buffers -- grown on first `upload_mesh`.
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid_vertex_buf"), size: 0,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solid_index_buf"), size: 0,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group,
            params_buf,
            shadow_pipeline,
            shadow_bind_group,
            shadow_map_view,
            msaa_color_view,
            depth_texture_view,
            vertex_buf,
            index_buf,
            vertex_capacity: 0,
            index_capacity:  0,
            index_count:     0,
        }
    }

    fn make_shadow_map_view(device: &wgpu::Device) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("solid_shadow_map_texture"),
            size:            wgpu::Extent3d { width: SHADOW_MAP_SIZE, height: SHADOW_MAP_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          DEPTH_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn make_msaa_color_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("solid_msaa_color_texture"),
            size:            wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    MSAA_SAMPLES,
            dimension:       wgpu::TextureDimension::D2,
            format:          CANVAS_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:    &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn make_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("solid_depth_texture"),
            size:            wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    MSAA_SAMPLES,
            dimension:       wgpu::TextureDimension::D2,
            format:          DEPTH_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:    &[],
        });
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Recreate the MSAA color + depth textures at new canvas dimensions.
    /// Independent of the vertex/index buffers and the shadow map (neither
    /// depends on canvas size).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.msaa_color_view = Self::make_msaa_color_view(device, width, height);
        self.depth_texture_view = Self::make_depth_view(device, width, height);
    }

    /// Upload a freshly-built CPU mesh, growing the GPU buffers only if the
    /// new mesh exceeds current capacity. Called only when the mesh is
    /// rebuilt (attractor type/params or Solid-specific sliders changed),
    /// never every frame.
    pub fn upload_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &sim::TubeMesh) {
        let vbytes = (mesh.vertices.len() * std::mem::size_of::<sim::TubeVertex>()) as u64;
        let ibytes = (mesh.indices.len() * std::mem::size_of::<u32>()) as u64;

        if vbytes > self.vertex_capacity {
            self.vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("solid_vertex_buf"), size: vbytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity = vbytes;
        }
        if ibytes > self.index_capacity {
            self.index_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("solid_index_buf"), size: ibytes,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.index_capacity = ibytes;
        }

        if vbytes > 0 {
            queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&mesh.vertices));
        }
        if ibytes > 0 {
            queue.write_buffer(&self.index_buf, 0, bytemuck::cast_slice(&mesh.indices));
        }
        self.index_count = mesh.indices.len() as u32;
    }

    /// Drop any uploaded mesh without rendering anything (used when the
    /// active attractor is detected as planar/2D and Solid mode is blocked).
    pub fn clear_mesh(&mut self) {
        self.index_count = 0;
    }

    /// Render the tube mesh into `canvas_view` with MSAA + depth testing +
    /// a shadow-mapped directional light, clearing with `bg_color` -- the
    /// Solid-mode replacement for dispatch_sim + composite + blit_to_canvas.
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        canvas_view: &wgpu::TextureView,
        params: SolidParams,
        bg_color: wgpu::Color,
    ) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        if self.index_count > 0 {
            // Pass 1: depth-only render from the light's point of view.
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("solid_shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes:    None,
                occlusion_query_set: None,
            });
            shadow_pass.set_pipeline(&self.shadow_pipeline);
            shadow_pass.set_bind_group(0, &self.shadow_bind_group, &[]);
            shadow_pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            shadow_pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            shadow_pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        // Pass 2: main lit + shadowed render into the MSAA color target.
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("solid_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           &self.msaa_color_view,
                resolve_target: Some(canvas_view),
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(bg_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
        });

        if self.index_count > 0 {
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.index_count, 0, 0..1);
        }
    }
}
