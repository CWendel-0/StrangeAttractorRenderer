// Compute shader: bilinear sub-pixel splatting into the 2× super-sampled histogram.
//
// Each trajectory point distributes its weight across the 4 surrounding super-sampled
// pixels in proportion to sub-pixel distance (bilinear interpolation).  This gives
// correct sub-pixel placement.  Spatial smoothing is handled separately in the
// composite shader by Density Estimation (DE).

struct SplatParams {
    view_proj: mat4x4<f32>,
    width:  u32,   // super-sampled width  = 2 × display width
    height: u32,   // super-sampled height = 2 × display height
    count:  u32,
    _pad:   u32,
}

struct InputPoint {
    pos:   vec3<f32>,
    speed: f32,
}

@group(0) @binding(0) var<storage, read>        points      : array<InputPoint>;
@group(0) @binding(1) var<storage, read_write>  accum       : array<atomic<u32>>;
@group(0) @binding(2) var<uniform>              params      : SplatParams;
@group(0) @binding(3) var<storage, read_write>  max_density : atomic<u32>;

const SPEED_SCALE:     f32 = 1024.0;
const MAX_SPEED_FIXED: f32 = 4294967040.0;
// Bilinear weights sum to WEIGHT_SCALE per point.
// The composite shader divides by WEIGHT_SCALE to recover fractional hit counts.
const WEIGHT_SCALE: u32 = 1024u;

fn splat_pixel(px: i32, py: i32, weight: u32, speed_fixed: u32) {
    if weight == 0u || px < 0 || py < 0 { return; }
    let upx = u32(px);
    let upy = u32(py);
    if upx >= params.width || upy >= params.height { return; }
    let base = (upy * params.width + upx) * 2u;
    let prev = atomicAdd(&accum[base], weight);
    atomicMax(&max_density, prev + weight);
    if speed_fixed > 0u {
        atomicAdd(&accum[base + 1u], speed_fixed);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }

    let pt = points[idx];

    let clip = params.view_proj * vec4<f32>(pt.pos, 1.0);
    if clip.w <= 0.0 { return; }

    let ndc = clip.xyz / clip.w;
    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < -1.0 || ndc.z > 1.0 { return; }

    let fx = (ndc.x * 0.5 + 0.5) * f32(params.width);
    let fy = (1.0 - (ndc.y * 0.5 + 0.5)) * f32(params.height);
    if fx < 0.0 || fy < 0.0 { return; }

    let px0 = i32(fx);
    let py0 = i32(fy);
    let dx  = fract(fx);
    let dy  = fract(fy);

    // Bilinear weights (sum ≈ WEIGHT_SCALE).
    let w00 = u32((1.0 - dx) * (1.0 - dy) * f32(WEIGHT_SCALE));
    let w10 = u32(       dx  * (1.0 - dy) * f32(WEIGHT_SCALE));
    let w01 = u32((1.0 - dx) *        dy  * f32(WEIGHT_SCALE));
    let w11 = u32(       dx  *        dy  * f32(WEIGHT_SCALE));

    let speed_fixed = u32(clamp(pt.speed * SPEED_SCALE, 0.0, MAX_SPEED_FIXED));

    splat_pixel(px0,     py0,     w00, speed_fixed);
    splat_pixel(px0 + 1, py0,     w10, 0u);
    splat_pixel(px0,     py0 + 1, w01, 0u);
    splat_pixel(px0 + 1, py0 + 1, w11, 0u);
}
