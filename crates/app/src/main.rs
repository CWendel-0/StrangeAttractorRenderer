mod camera;
mod gradient;
mod ui;
mod velocity_slider;

use camera::ArcballCamera;
use ui::UiState;

use gpu::histogram::{CompositeParams, Histogram};
use sim::{AttractorConfig, AttractorType, LyapunovWorker, SearchWorker};

use glam::Vec2;
use pollster::block_on;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

// ---- GPU state -------------------------------------------------------------

struct GpuState {
    surface:       wgpu::Surface<'static>,
    device:        Arc<wgpu::Device>,
    queue:         Arc<wgpu::Queue>,
    config:        wgpu::SurfaceConfiguration,
    histogram:     Histogram,
    egui_renderer: egui_wgpu::Renderer,
}

impl GpuState {
    async fn new(window: Arc<Window>, initial_config: &AttractorConfig) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits {
                        max_buffer_size:                  adapter.limits().max_buffer_size,
                        max_storage_buffer_binding_size:  adapter.limits().max_storage_buffer_binding_size,
                        ..wgpu::Limits::default()
                    },
                    memory_hints:      Default::default(),
                },
                None,
            )
            .await
            .expect("device creation failed");

        let caps   = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width:        size.width,
            height:       size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let histogram = Histogram::new(
            &device, &queue, size.width, size.height, format, initial_config,
        );

        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        GpuState { surface, device, queue, config, histogram, egui_renderer }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.config.width  = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.histogram.resize(&self.device, new_size.width, new_size.height);
    }
}

// ---- Application -----------------------------------------------------------

struct App {
    window:    Option<Arc<Window>>,
    gpu:       Option<GpuState>,

    // Camera
    camera:    ArcballCamera,

    // Input state
    mouse_pos:   Vec2,
    mouse_left:  bool,
    mouse_mid:   bool,

    // UI
    egui_ctx:   egui::Context,
    egui_state: Option<egui_winit::State>,
    ui:         Option<UiState>,

    pending_clear:  bool,
    search_worker:  Option<SearchWorker>,
    searching:      bool,

    lyapunov_worker:  Option<LyapunovWorker>,
    lyapunov_metrics: Option<(f32, f32)>,

    frames_accumulated: u32,
    reseed_count:       u32,
}

impl App {
    fn new() -> Self {
        Self {
            window:    None,
            gpu:       None,
            camera:    ArcballCamera::new(1.0),
            mouse_pos: Vec2::ZERO,
            mouse_left: false,
            mouse_mid:  false,
            egui_ctx:   egui::Context::default(),
            egui_state: None,
            ui:         None,
            pending_clear: false,
            search_worker: None,
            searching:     false,
            lyapunov_worker:  None,
            lyapunov_metrics: None,
            frames_accumulated: 0,
            reseed_count:       0,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("Strange Attractor")
            .with_inner_size(PhysicalSize::new(1280u32, 800u32));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let init_config = AttractorConfig::new(AttractorType::default());
        let gpu = block_on(GpuState::new(Arc::clone(&window), &init_config));

        let size   = window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let (bb_min, bb_max) = init_config.estimate_bounds_cpu();
        self.camera = ArcballCamera::fit_aabb(aspect, bb_min, bb_max);

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );

        self.ui = Some(UiState::new());

        self.window     = Some(window);
        self.gpu        = Some(gpu);
        self.egui_state = Some(egui_state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        if let (Some(state), Some(window)) = (&mut self.egui_state, &self.window) {
            let resp = state.on_window_event(window, &event);
            if resp.consumed { return; }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size);
                    self.camera.resize(size.width, size.height);
                }
                let actual_ss = self.gpu.as_ref().map(|g| g.histogram.ss_scale);
                if let (Some(ss), Some(ui)) = (actual_ss, self.ui.as_mut()) {
                    ui.ss_scale = ss;
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                match button {
                    MouseButton::Left   => self.mouse_left = state == ElementState::Pressed,
                    MouseButton::Middle => self.mouse_mid  = state == ElementState::Pressed,
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = Vec2::new(position.x as f32, position.y as f32);
                let delta   = new_pos - self.mouse_pos;
                self.mouse_pos = new_pos;

                if let Some(window) = &self.window {
                    let s = window.inner_size();
                    let vp = Vec2::new(s.width as f32, s.height as f32);
                    if self.mouse_left { self.camera.rotate(delta, vp); }
                    if self.mouse_mid  { self.camera.pan(delta, vp); }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y)  => y,
                    MouseScrollDelta::PixelDelta(p)    => p.y as f32 * 0.01,
                };
                let factor = if scroll > 0.0 { 0.9 } else { 1.0 / 0.9 };
                self.camera.zoom(factor);
            }

            WindowEvent::RedrawRequested => {
                self.render();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl App {
    fn render(&mut self) {
        let (gpu, ui, egui_state, window) = match (
            self.gpu.as_mut(),
            self.ui.as_mut(),
            self.egui_state.as_mut(),
            self.window.as_ref(),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return,
        };

        // ---- Search result handling ----
        // Check before building the UI so button state is up to date this frame.
        let mut search_result_received = false;
        if let Some(worker) = &self.search_worker {
            if let Ok(result) = worker.result_rx.try_recv() {
                // Only apply if the user hasn't switched to a different attractor type.
                if result.attractor_type == ui.attractor.attractor_type {
                    ui.attractor.params = result.params;
                    // Call reset_sim_states directly — ui.show() clears ui.dirty before
                    // the if-ui.dirty check runs, so we can't route through that flag.
                    gpu.histogram.reset_sim_states(&gpu.queue, &ui.attractor);

                    let size   = window.inner_size();
                    let aspect = size.width as f32 / size.height.max(1) as f32;
                    self.camera = ArcballCamera::fit_aabb(aspect, result.bb_min, result.bb_max);
                    self.pending_clear = true;
                    search_result_received = true;
                }
                self.search_worker = None;
                self.searching = false;
            }
        }

        // ---- egui input + UI ----
        let raw_input = egui_state.take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);
        ui.show(&self.egui_ctx, self.searching, self.lyapunov_metrics);
        let full_output = self.egui_ctx.end_pass();
        egui_state.handle_platform_output(window, full_output.platform_output.clone());

        // ---- Search request handling ----
        if ui.search_requested {
            // Cancel any running search and start a fresh one.
            self.search_worker = None;
            self.searching = true;
            self.search_worker = Some(SearchWorker::spawn(ui.attractor.clone()));
        }

        if ui.dirty {
            gpu.histogram.reset_sim_states(&gpu.queue, &ui.attractor);
            self.pending_clear = true;
        }

        if ui.type_changed {
            // Cancel any search for the old type.
            self.search_worker = None;
            self.searching = false;

            let size = window.inner_size();
            let aspect = size.width as f32 / size.height.max(1) as f32;
            let (bb_min, bb_max) = ui.attractor.estimate_bounds_cpu();
            self.camera = ArcballCamera::fit_aabb(aspect, bb_min, bb_max);
        }

        // ---- Lyapunov worker management ----
        // Poll for a completed result.
        if let Some(worker) = &self.lyapunov_worker {
            match worker.result_rx.try_recv() {
                Ok(v) => {
                    self.lyapunov_metrics = Some(v);
                    self.lyapunov_worker = None;
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    // Thread exited without finding valid metrics.
                    self.lyapunov_metrics = Some((f32::NAN, f32::NAN));
                    self.lyapunov_worker = None;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {}
            }
        }
        // Clear stale display when the attractor type changes.
        if ui.type_changed {
            self.lyapunov_metrics = None;
        }
        // (Re-)spawn whenever params change, a search result landed, or on initial launch.
        if ui.dirty || ui.type_changed || search_result_received
            || (self.lyapunov_worker.is_none() && self.lyapunov_metrics.is_none())
        {
            self.lyapunov_worker = Some(LyapunovWorker::spawn(ui.attractor.clone()));
        }

        if ui.gradient_a_dirty {
            gpu.histogram.upload_gradient_a(&gpu.queue, &ui.gradient_a.to_rgba8());
            ui.gradient_a_dirty = false;
        }
        if ui.gradient_b_dirty {
            gpu.histogram.upload_gradient_b(&gpu.queue, &ui.gradient_b.to_rgba8());
            ui.gradient_b_dirty = false;
        }

        if self.camera.dirty {
            gpu.histogram.reset_sim_states(&gpu.queue, &ui.attractor);
            self.pending_clear = true;
            self.camera.dirty = false;
        }

        if ui.ss_scale != gpu.histogram.ss_scale {
            gpu.histogram.set_ss_scale(&gpu.device, ui.ss_scale);
            ui.ss_scale = gpu.histogram.ss_scale;
            self.pending_clear = true;
        }

        gpu.histogram.poll_max_density(&gpu.device);

        // Periodic reseed: every ~30 s inject a new set of 8 192 uniformly-sampled
        // positions into the live trajectories without clearing the accumulator.
        // Each reseed uses a different phase offset so the new positions are always
        // distinct from the previous batch (512 × phase >> Lyapunov time for all
        // built-in attractors, giving statistically independent samples).
        const RESEED_INTERVAL: u32 = 200; // ~30 s at 60 fps
        self.frames_accumulated = if self.pending_clear { 0 } else { self.frames_accumulated + 1 };
        if self.frames_accumulated > 0 && self.frames_accumulated % RESEED_INTERVAL == 0 {
            gpu.histogram.reseed_trajectories(&gpu.queue, &ui.attractor, self.reseed_count);
            self.reseed_count = self.reseed_count.wrapping_add(1);
        }

        // ---- GPU frame ----
        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            Err(_) => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        if self.pending_clear {
            gpu.histogram.clear(&mut encoder);
            self.pending_clear = false;
        }

        let vp = self.camera.view_proj();
        gpu.histogram.dispatch_sim(&gpu.queue, &mut encoder, vp, &ui.attractor);

        let w = gpu.config.width;
        let h = gpu.config.height;
        const WEIGHT_SCALE: f32 = 1024.0;
        let ss = gpu.histogram.ss_scale;
        let max_display = gpu.histogram.last_max_density as f32 / WEIGHT_SCALE * (ss * ss) as f32;
        let log_max = (max_display + 1.0).ln().max(1e-6);
        gpu.histogram.composite(
            &gpu.queue,
            &mut encoder,
            CompositeParams {
                width:           w,
                height:          h,
                log_max_density: log_max,
                brightness:      ui.brightness,
                gamma:           ui.gamma,
                ss_width:        w * ss,
                ss_height:       h * ss,
                max_sigma:       ui.max_sigma,
                min_sigma:       ui.min_sigma,
                ss_scale:        ss,
                blend_mode:      ui.blend_mode.as_u32(),
                max_speed_enc:   gpu.histogram.last_max_speed,
            },
            ui.render_mode,
        );

        gpu.histogram.blit(&mut encoder, &view);
        gpu.histogram.encode_max_readback(&mut encoder);

        // ---- egui render pass ----
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels:    [w, h],
            pixels_per_point:  window.scale_factor() as f32,
        };
        let tris = self.egui_ctx.tessellate(full_output.shapes, screen_desc.pixels_per_point);
        for (id, delta) in full_output.textures_delta.set {
            gpu.egui_renderer.update_texture(&gpu.device, &gpu.queue, id, &delta);
        }
        gpu.egui_renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &tris, &screen_desc);

        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label:                    Some("egui_pass"),
                    color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                        view:           &view,
                        resolve_target: None,
                        ops:            wgpu::Operations {
                            load:  wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes:         None,
                    occlusion_query_set:      None,
                })
                .forget_lifetime();
            gpu.egui_renderer.render(&mut rpass, &tris, &screen_desc);
        }

        for id in &full_output.textures_delta.free {
            gpu.egui_renderer.free_texture(id);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        gpu.histogram.submit_max_readback();
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
