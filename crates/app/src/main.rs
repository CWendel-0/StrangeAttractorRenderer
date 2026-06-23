mod camera;
mod colorset;
mod gradient;
mod movie;
mod state;
mod ui;
mod velocity_slider;

use camera::ArcballCamera;
use state::SceneState;
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
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Base multiplicative zoom step per unit of scroll. Scaling by the actual
// scroll magnitude (instead of just its sign) keeps a single wheel "click"
// gentle while still letting fast/heavy scrolling zoom further in one event.
const ZOOM_STEP: f32 = 0.95;

fn zoom_factor_from_scroll(scroll: f32) -> f32 {
    ZOOM_STEP.powf(scroll.clamp(-5.0, 5.0))
}

/// Applies the linear→sRGB transfer function to the RGB channels of an RGBA8
/// buffer in place, matching what egui's sRGB display path already does, so
/// saved images match what's on screen.
fn srgb_encode_pixels(pixels: &mut [u8]) {
    for chunk in pixels.chunks_mut(4) {
        for ch in &mut chunk[0..3] {
            let f = *ch as f32 / 255.0;
            let s = if f <= 0.003_130_8 {
                f * 12.92
            } else {
                1.055 * f.powf(1.0 / 2.4) - 0.055
            };
            *ch = (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
}

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
    async fn new(window: Arc<Window>, initial_config: &AttractorConfig, canvas_w: u32, canvas_h: u32) -> Self {
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
            &device, &queue, canvas_w, canvas_h, initial_config,
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
    keys_down:   std::collections::HashSet<KeyCode>,

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

    canvas_tex_id:      Option<egui::TextureId>, // Rgba8Unorm view  → egui shows sRGB
    canvas_tex_id_raw:  Option<egui::TextureId>, // Rgba8UnormSrgb view → egui shows linear

    movie_job: Option<movie::MovieJob>,
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
            keys_down:  std::collections::HashSet::new(),
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
            canvas_tex_id:     None,
            canvas_tex_id_raw: None,
            movie_job: None,
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

        let size     = window.inner_size();
        let canvas_w = (size.width  / 2).max(64);
        let canvas_h = (size.height / 2).max(64);

        let init_config = AttractorConfig::new(AttractorType::default());
        let mut gpu = block_on(GpuState::new(Arc::clone(&window), &init_config, canvas_w, canvas_h));

        let aspect = canvas_w as f32 / canvas_h.max(1) as f32;

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

        let canvas_tex_id = gpu.egui_renderer.register_native_texture(
            &gpu.device,
            gpu.histogram.canvas_view(),
            wgpu::FilterMode::Linear,
        );
        let canvas_tex_id_raw = gpu.egui_renderer.register_native_texture(
            &gpu.device,
            gpu.histogram.canvas_view_srgb(),
            wgpu::FilterMode::Linear,
        );
        self.canvas_tex_id     = Some(canvas_tex_id);
        self.canvas_tex_id_raw = Some(canvas_tex_id_raw);

        self.ui = Some(UiState::new(canvas_w, canvas_h, colorset::load_custom_color_sets()));

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
            if resp.consumed {
                // Button releases must still clear camera flags even when egui
                // consumed the event — otherwise the release is silently dropped
                // and the camera stays locked in rotate/pan mode indefinitely.
                if let WindowEvent::MouseInput { state: ElementState::Released, button, .. } = &event {
                    match button {
                        MouseButton::Left   => self.mouse_left = false,
                        MouseButton::Middle => self.mouse_mid  = false,
                        _ => {}
                    }
                }
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size);
                    // canvas dimensions are user-controlled, not tied to window size
                }
            }

            WindowEvent::Focused(false) => {
                // Drop held keys on focus loss — a key released while the window
                // wasn't focused would otherwise appear stuck down forever.
                self.keys_down.clear();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed  => { self.keys_down.insert(code); }
                        ElementState::Released => { self.keys_down.remove(&code); }
                    }
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
                self.camera.zoom(zoom_factor_from_scroll(scroll));
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

                    let cw = gpu.histogram.width;
                    let ch = gpu.histogram.height;
                    let aspect = cw as f32 / ch.max(1) as f32;
                    self.camera = ArcballCamera::fit_aabb(aspect, result.bb_min, result.bb_max);
                    self.pending_clear = true;
                    search_result_received = true;
                }
                self.search_worker = None;
                self.searching = false;
            }
        }

        // ---- egui input + UI ----
        ui.movie_job_active = self.movie_job.is_some();
        let raw_input = egui_state.take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);
        let active_tex_id = if ui.color_space_srgb { self.canvas_tex_id } else { self.canvas_tex_id_raw };
        ui.show(&self.egui_ctx, self.searching, self.lyapunov_metrics, active_tex_id);
        let full_output = self.egui_ctx.end_pass();
        egui_state.handle_platform_output(window, full_output.platform_output.clone());

        // ---- Canvas viewport input (drag/scroll forwarded from the Image widget) ----
        {
            let vp = Vec2::new(gpu.histogram.width as f32, gpu.histogram.height as f32);
            if ui.viewport_drag_left != egui::Vec2::ZERO {
                let d = ui.viewport_drag_left;
                self.camera.rotate(Vec2::new(d.x, d.y), vp);
            }
            if ui.viewport_drag_middle != egui::Vec2::ZERO {
                let d = ui.viewport_drag_middle;
                self.camera.pan(Vec2::new(d.x, d.y), vp);
            }
            if ui.viewport_scroll != 0.0 {
                self.camera.zoom(zoom_factor_from_scroll(ui.viewport_scroll));
            }
        }

        // ---- Keyboard camera controls: WASD rotate, arrows pan, +/- zoom ----
        // Skipped while a UI widget wants keyboard focus (e.g. editing a DragValue),
        // so typing into a parameter field doesn't also spin the camera.
        if !self.egui_ctx.wants_keyboard_input() {
            let vp = Vec2::new(gpu.histogram.width as f32, gpu.histogram.height as f32);
            const ROTATE_SPEED: f32 = 6.0;
            const PAN_SPEED:    f32 = 6.0;

            let mut rotate_delta = Vec2::ZERO;
            if self.keys_down.contains(&KeyCode::KeyW) { rotate_delta.y -= ROTATE_SPEED; }
            if self.keys_down.contains(&KeyCode::KeyS) { rotate_delta.y += ROTATE_SPEED; }
            if self.keys_down.contains(&KeyCode::KeyA) { rotate_delta.x -= ROTATE_SPEED; }
            if self.keys_down.contains(&KeyCode::KeyD) { rotate_delta.x += ROTATE_SPEED; }
            if rotate_delta != Vec2::ZERO { self.camera.rotate(rotate_delta, vp); }

            let mut pan_delta = Vec2::ZERO;
            if self.keys_down.contains(&KeyCode::ArrowUp)    { pan_delta.y -= PAN_SPEED; }
            if self.keys_down.contains(&KeyCode::ArrowDown)  { pan_delta.y += PAN_SPEED; }
            if self.keys_down.contains(&KeyCode::ArrowLeft)  { pan_delta.x -= PAN_SPEED; }
            if self.keys_down.contains(&KeyCode::ArrowRight) { pan_delta.x += PAN_SPEED; }
            if pan_delta != Vec2::ZERO { self.camera.pan(pan_delta, vp); }

            let zoom_in  = self.keys_down.contains(&KeyCode::Equal) || self.keys_down.contains(&KeyCode::NumpadAdd);
            let zoom_out = self.keys_down.contains(&KeyCode::Minus) || self.keys_down.contains(&KeyCode::NumpadSubtract);
            if zoom_in  { self.camera.zoom(zoom_factor_from_scroll(1.0)); }
            if zoom_out { self.camera.zoom(zoom_factor_from_scroll(-1.0)); }
        }

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

            let cw = gpu.histogram.width;
            let ch = gpu.histogram.height;
            let aspect = cw as f32 / ch.max(1) as f32;
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

        // ---- Movie job: cancel / start the next interpolated frame ----
        if let Some(job) = &mut self.movie_job {
            if ui.movie_cancel_requested {
                job.cancel();
            }
            if matches!(job.status(), movie::MovieStatus::Rendering { .. }) && !job.frame_started() {
                let interpolated = job.current_interpolated_state();
                interpolated.apply(ui, &mut self.camera);
                gpu.histogram.reset_sim_states(&gpu.queue, &ui.attractor);
                self.pending_clear = true;
                job.mark_frame_started();
            }
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
            // Only the screen-space projection changed — the attractor's 3D
            // trajectories are unaffected and already well-distributed across
            // it, so just clear the (now stale, projection-dependent) histogram
            // and keep simulating the existing states under the new view.
            // Re-seeding here would otherwise restart every trajectory from the
            // same deterministic positions on every drag frame, which never lets
            // the density accumulate and biases the picture toward whatever the
            // fixed initial seed happens to hit first.
            self.pending_clear = true;
            self.camera.dirty = false;
        }

        if ui.ss_scale != gpu.histogram.ss_scale {
            gpu.histogram.set_ss_scale(&gpu.device, ui.ss_scale);
            ui.ss_scale = gpu.histogram.ss_scale;
            self.pending_clear = true;
        }

        if ui.canvas_dirty {
            let cw = ui.canvas_width.max(64);
            let ch = ui.canvas_height.max(64);
            gpu.histogram.resize(&gpu.device, cw, ch);
            if let Some(id) = self.canvas_tex_id {
                gpu.egui_renderer.update_egui_texture_from_wgpu_texture(
                    &gpu.device,
                    gpu.histogram.canvas_view(),
                    wgpu::FilterMode::Linear,
                    id,
                );
            }
            if let Some(id) = self.canvas_tex_id_raw {
                gpu.egui_renderer.update_egui_texture_from_wgpu_texture(
                    &gpu.device,
                    gpu.histogram.canvas_view_srgb(),
                    wgpu::FilterMode::Linear,
                    id,
                );
            }
            self.camera.resize(cw, ch);
            self.pending_clear = true;
            ui.canvas_dirty = false;
        }

        gpu.histogram.poll_max_density(&gpu.device);

        // Periodic reseed: every ~30 s inject a new set of 8 192 uniformly-sampled
        // positions into the live trajectories without clearing the accumulator.
        // Each reseed uses a different phase offset so the new positions are always
        // distinct from the previous batch (512 × phase >> Lyapunov time for all
        // built-in attractors, giving statistically independent samples).
        const RESEED_INTERVAL: u32 = 200; // ~30 s at 60 fps
        if self.movie_job.is_none() {
            self.frames_accumulated = if self.pending_clear { 0 } else { self.frames_accumulated + 1 };
            if self.frames_accumulated > 0 && self.frames_accumulated % RESEED_INTERVAL == 0 {
                gpu.histogram.reseed_trajectories(&gpu.queue, &ui.attractor, self.reseed_count);
                self.reseed_count = self.reseed_count.wrapping_add(1);
            }
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
            ui.iter_count = 0;
        }

        let vp = self.camera.view_proj();
        gpu.histogram.dispatch_sim(&gpu.queue, &mut encoder, vp, &ui.attractor, ui.noise_magnitude);
        ui.iter_count = ui.iter_count.saturating_add(
            gpu::histogram::SIM_NUM_TRAJECTORIES as u64 * gpu::histogram::SIM_STEPS_PER_DISPATCH as u64
        );

        let cw = gpu.histogram.width;
        let ch = gpu.histogram.height;
        const WEIGHT_SCALE: f32 = 1024.0;
        let ss = gpu.histogram.ss_scale;
        let max_display = gpu.histogram.last_max_density as f32 / WEIGHT_SCALE;
        let log_max = (max_display + 1.0).ln().max(1e-6);
        gpu.histogram.composite(
            &gpu.queue,
            &mut encoder,
            CompositeParams {
                width:           cw,
                height:          ch,
                log_max_density: log_max,
                brightness:      ui.brightness,
                gamma:           ui.gamma.max(1e-4),
                ss_width:        cw * ss,
                ss_height:       ch * ss,
                max_sigma:       ui.max_sigma,
                min_sigma:       ui.min_sigma,
                ss_scale:        ss,
                blend_mode:      ui.blend_mode.as_u32(),
                max_speed_enc:   gpu.histogram.last_max_speed,
                alpha_power:     ui.alpha_power,
            },
            ui.render_mode,
            {
                let c = ui.bg_color;
                wgpu::Color {
                    r: c.r() as f64 / 255.0,
                    g: c.g() as f64 / 255.0,
                    b: c.b() as f64 / 255.0,
                    a: 1.0,
                }
            },
        );

        gpu.histogram.blit_to_canvas(&mut encoder);
        gpu.histogram.encode_max_readback(&mut encoder);

        // ---- egui render pass ----
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels:    [gpu.config.width, gpu.config.height],
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
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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

        // ---- Movie job: per-frame completion ----
        if let Some(job) = &mut self.movie_job {
            if let movie::MovieStatus::Rendering { .. } = job.status() {
                if job.iter_target_reached(ui.iter_count) {
                    let (w, h, mut pixels) = gpu.histogram.read_canvas_pixels(&gpu.device, &gpu.queue);
                    if ui.color_space_srgb {
                        srgb_encode_pixels(&mut pixels);
                    }
                    let path = job.frame_filename();
                    if let Err(e) = image::save_buffer(&path, &pixels, w, h, image::ColorType::Rgba8) {
                        job.fail(format!("Failed to write frame {}: {e}", path.display()));
                    } else if job.advance_frame() {
                        job.run_encode();
                    }
                }
            }
            ui.movie_status_for_ui = Some(job.status().clone());
        }

        if ui.save_requested {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let default_name = format!("attractor_{ts}.png");
            // Blocks the event loop until the user picks a path or cancels —
            // acceptable here since this only runs on an explicit user action.
            let chosen_path = rfd::FileDialog::new()
                .set_title("Save Image")
                .set_file_name(&default_name)
                .add_filter("PNG Image", &["png"])
                .save_file();

            if let Some(path) = chosen_path {
                let (w, h, mut pixels) = gpu.histogram.read_canvas_pixels(&gpu.device, &gpu.queue);
                // In sRGB mode egui applies a linear→sRGB transfer before the display,
                // so apply the same encoding to RGB channels before saving.
                // In linear mode the raw values are saved as-is to match the display.
                if ui.color_space_srgb {
                    srgb_encode_pixels(&mut pixels);
                }
                match image::save_buffer(&path, &pixels, w, h, image::ColorType::Rgba8) {
                    Ok(()) => log::info!("Saved {}", path.display()),
                    Err(e) => log::error!("Failed to save PNG: {e}"),
                }
            }
        }

        if ui.save_state_requested {
            let chosen_path = rfd::FileDialog::new()
                .set_title("Save State")
                .set_file_name("attractor_state.json")
                .add_filter("Attractor State", &["json"])
                .save_file();
            if let Some(path) = chosen_path {
                let state = SceneState::capture(ui, &self.camera);
                match state.save_to_file(&path) {
                    Ok(()) => log::info!("Saved state to {}", path.display()),
                    Err(e) => log::error!("Failed to save state: {e}"),
                }
            }
        }

        if ui.load_state_requested {
            let chosen_path = rfd::FileDialog::new()
                .set_title("Load State")
                .add_filter("Attractor State", &["json"])
                .pick_file();
            if let Some(path) = chosen_path {
                match SceneState::load_from_file(&path) {
                    Ok(loaded) => {
                        loaded.apply(ui, &mut self.camera);
                        // Re-render from scratch with the loaded settings — the
                        // histogram/image are intentionally not part of a saved
                        // state, only the settings needed to reproduce it.
                        gpu.histogram.reset_sim_states(&gpu.queue, &ui.attractor);
                        self.pending_clear = true;
                        self.search_worker = None;
                        self.searching = false;
                        self.lyapunov_metrics = None;
                        self.lyapunov_worker = Some(LyapunovWorker::spawn(ui.attractor.clone()));
                        log::info!("Loaded state from {}", path.display());
                    }
                    Err(e) => log::error!("Failed to load state: {e}"),
                }
            }
        }

        if ui.movie_render_requested {
            let mut keyframes = Vec::with_capacity(ui.movie_keyframe_paths.len());
            let mut load_error = None;
            for path in &ui.movie_keyframe_paths {
                match SceneState::load_from_file(path) {
                    Ok(s) => keyframes.push(s),
                    Err(e) => {
                        load_error = Some(format!("Failed to load {}: {e}", path.display()));
                        break;
                    }
                }
            }
            match load_error {
                Some(e) => ui.movie_status_for_ui = Some(movie::MovieStatus::Error(e)),
                None => {
                    let settings = movie::MovieSettings {
                        frames_per_step: ui.movie_frames_per_step,
                        iters_per_frame: ui.movie_iters_per_frame,
                        loop_back:       ui.movie_loop_back,
                        output_kind:     ui.movie_output_kind,
                        fps:             ui.movie_fps,
                        mp4_crf:         ui.movie_mp4_crf,
                    };
                    let output_path = ui.movie_output_path.clone().unwrap_or_default();
                    match movie::MovieJob::new(keyframes, settings, output_path) {
                        Ok(job) => self.movie_job = Some(job),
                        Err(e) => ui.movie_status_for_ui = Some(movie::MovieStatus::Error(e)),
                    }
                }
            }
        }

        if ui.movie_close_requested {
            self.movie_job = None;
            ui.movie_status_for_ui = None;
        }

        if ui.color_sets_dirty {
            colorset::save_custom_color_sets(&ui.color_sets_custom);
            ui.color_sets_dirty = false;
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
