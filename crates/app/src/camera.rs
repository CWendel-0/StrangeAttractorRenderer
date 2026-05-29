use glam::{Mat4, Quat, Vec2, Vec3};

pub struct ArcballCamera {
    target:   Vec3,
    distance: f32,
    rotation: Quat,
    aspect:   f32,
    fov_y:    f32,
    pub dirty: bool,
}

impl ArcballCamera {
    pub fn new(aspect: f32) -> Self {
        Self {
            target:   Vec3::ZERO,
            distance: 80.0,
            rotation: Quat::IDENTITY,
            aspect,
            fov_y: 45_f32.to_radians(),
            dirty: false,
        }
    }

    /// Construct a camera whose frustum just contains a bounding sphere derived
    /// from `bb_min`/`bb_max`, with 10 % margin.  The target is the AABB centre.
    pub fn fit_aabb(aspect: f32, bb_min: Vec3, bb_max: Vec3) -> Self {
        let fov_y    = 45_f32.to_radians();
        let fov_x    = 2.0 * ((fov_y * 0.5).tan() * aspect).atan();
        let fov_min  = fov_y.min(fov_x);
        let center   = (bb_min + bb_max) * 0.5;
        let half_diag = (bb_max - bb_min).length() * 0.5;
        let distance = half_diag / (fov_min * 0.5).tan() * 1.1;
        Self {
            target:   center,
            distance: distance.clamp(1.0, 5000.0),
            rotation: Quat::IDENTITY,
            aspect,
            fov_y,
            dirty: false,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }

    /// Rotate by a screen-space drag delta (pixels).
    pub fn rotate(&mut self, delta: Vec2, viewport: Vec2) {
        if viewport.x == 0.0 || viewport.y == 0.0 { return; }
        let scale = std::f32::consts::PI / viewport.x.min(viewport.y);
        let yaw   = Quat::from_rotation_y(-delta.x * scale);
        let pitch = Quat::from_rotation_x(-delta.y * scale);
        self.rotation = (yaw * self.rotation * pitch).normalize();
        self.dirty = true;
    }

    /// Pan in the camera's local XY plane.
    pub fn pan(&mut self, delta: Vec2, viewport: Vec2) {
        if viewport.x == 0.0 || viewport.y == 0.0 { return; }
        let scale = self.distance / viewport.x.min(viewport.y);
        let right = self.rotation * Vec3::X;
        let up    = self.rotation * Vec3::Y;
        self.target -= right * delta.x * scale;
        self.target += up   * delta.y * scale;
        self.dirty = true;
    }

    /// Zoom by multiplicative factor.
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(1.0, 5000.0);
        self.dirty = true;
    }

    pub fn view_proj(&self) -> Mat4 {
        let eye = self.target + self.rotation * Vec3::new(0.0, 0.0, self.distance);
        let view = Mat4::look_at_rh(eye, self.target, self.rotation * Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect, 0.1, 10_000.0);
        proj * view
    }
}
