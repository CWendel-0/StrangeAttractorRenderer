use glam::Vec3;

/// A single simulated point with its instantaneous speed.
///
/// `#[repr(C)]` + Pod ensures the layout matches the WGSL `InputPoint` struct
/// (vec3<f32> at offset 0, f32 at offset 12, stride 16), allowing direct
/// `bytemuck::cast_slice` upload without an intermediate repack.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Point {
    pub pos: Vec3,
    pub speed: f32,
}

/// Metadata for one parameter exposed to the UI.
#[derive(Clone, Debug)]
pub struct ParamDesc {
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

/// Sample `steps` trajectory points and return the axis-aligned bounding box
/// `(min, max)` of the positions visited.  Falls back to a unit cube at the
/// origin if the trajectory produces no finite points.
///
/// The attractor is reset to a fresh initial condition after sampling so the
/// caller can hand it to a SimWorker without residual state.
pub fn estimate_bounds<A: Attractor>(attractor: &mut A, steps: usize) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for _ in 0..steps {
        if let Some(pt) = attractor.step() {
            if pt.pos.is_finite() {
                min = min.min(pt.pos);
                max = max.max(pt.pos);
            }
        }
    }
    let params = attractor.params().to_vec();
    attractor.reset(&params);
    if min.x > max.x {
        (Vec3::splat(-1.0), Vec3::splat(1.0))
    } else {
        (min, max)
    }
}

/// Core simulation trait.  All concrete attractors implement this.
pub trait Attractor: Send + 'static {
    /// Advance one integration step.
    /// Returns `Some((position, speed))` or `None` if the orbit has escaped.
    fn step(&mut self) -> Option<Point>;

    /// Reset to a fresh initial condition using the provided parameters.
    fn reset(&mut self, params: &[f32]);

    /// Static descriptor list for the UI (name, range, default).
    fn param_descriptors() -> &'static [ParamDesc]
    where
        Self: Sized;

    /// Current parameter values (same order as `param_descriptors`).
    fn params(&self) -> &[f32];
}
