// GPU-side Chen-Celikovsky attractor simulation.
//
// Params in p[]:
//   p[0].x = a, p[0].y = b, p[0].z = c, p[0].w = dt
//
// Equations:
//   x' = x + dt*a*(y-x)
//   y' = y + dt*(-x*z + c*y)
//   z' = z + dt*(x*y - b*z)

struct SimParams {
    view_proj: mat4x4<f32>,
    ss_width:  u32,
    ss_height: u32,
    steps:     u32,
    num_traj:  u32,
    light_view_proj: mat4x4<f32>,      // orthographic, for Points mode shadow-buffer splatting
    points_radius:   u32,              // Points mode camera-space splat footprint radius (0 = single pixel)
    light_buf_size:  u32,              // Points mode shadow buffer resolution (square)
    _points_pad:     vec2<u32>,        // keeps `p`'s start offset 16-byte aligned
    p:         array<vec4<f32>, 12>,
}

@group(0) @binding(0) var<storage, read_write> states:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accum:       array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> max_vals: array<atomic<u32>, 2>;
@group(0) @binding(3) var<uniform>             params:      SimParams;
@group(0) @binding(4) var<storage, read_write> points_depth: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> points_hit: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> points_light_depth: array<atomic<u32>>;

const WEIGHT_SCALE: f32 = 1024.0;
const SPEED_SCALE:  u32 = 256u;

fn splat_pixel(px: i32, py: i32, weight: u32, speed_contrib: u32) {
    if weight == 0u || px < 0 || py < 0 { return; }
    let upx = u32(px);
    let upy = u32(py);
    if upx >= params.ss_width || upy >= params.ss_height { return; }
    let base = (upy * params.ss_width + upx) * 2u;
    if atomicLoad(&accum[base]) < 0x7FFFFFFFu {
        let prev = atomicAdd(&accum[base], weight);
        atomicMax(&max_vals[0], prev + weight);
    }
    if atomicLoad(&accum[base + 1u]) < 0x7FFFFFFFu {
        atomicAdd(&accum[base + 1u], speed_contrib);
    }
}

fn recovery_hash(seed: u32) -> f32 {
    var h = seed;
    h ^= h >> 16u; h *= 0x7feb352du;
    h ^= h >> 15u; h *= 0x846ca68bu;
    h ^= h >> 16u;
    return f32(h) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let traj = gid.x;
    if traj >= params.num_traj { return; }

    var x = states[traj].x;
    var y = states[traj].y;
    var z = states[traj].z;
    var rng = bitcast<u32>(states[traj].w);
    if rng == 0u { rng = traj + 1u; }

    let a  = params.p[0].x;
    let b  = params.p[0].y;
    let c  = params.p[0].z;
    let dt = params.p[0].w;

    for (var i = 0u; i < params.steps; i++) {
        let vx = a*(y-x);
        let vy = -(x*z) + c*y;
        let vz = x*y - b*z;

        let nx = x + dt * vx;
        let ny = y + dt * vy;
        let nz = z + dt * vz;
        x = nx; y = ny; z = nz;

        if abs(x) > 1e6 || abs(y) > 1e6 || abs(z) > 1e6 {
            x = 1.0 + (recovery_hash(traj) - 0.5) * 0.2;
            y = 1.0 + (recovery_hash(traj ^ 0x9E3779B9u) - 0.5) * 0.2;
            z = 1.0 + (recovery_hash(traj ^ 0x85EBCA6Bu) - 0.5) * 0.2;
            for (var w = 0u; w < 500u; w++) {
                let wx = x; let wy = y; let wz = z;
                x = wx + dt * (a*(wy-wx));
                y = wy + dt * (-(wx*wz) + c*wy);
                z = wz + dt * (wx*wy - b*wz);
            }
            continue;
        }

        let speed_raw = length(vec3<f32>(vx, vy, vz));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);
        atomicMax(&max_vals[1], speed_enc);

        let world_pos4 = vec4<f32>(x, y, z, 1.0);

        // Light-space splat first, independent of camera visibility -- the
        // shadow buffer should reflect points even when they're currently
        // outside the camera's view.
        let light_clip = params.light_view_proj * world_pos4;
        if light_clip.w > 0.0 {
            let light_ndc = light_clip.xyz / light_clip.w;
            if light_ndc.x >= -1.0 && light_ndc.x <= 1.0 && light_ndc.y >= -1.0 && light_ndc.y <= 1.0 {
                let light_fx = (light_ndc.x * 0.5 + 0.5) * f32(params.light_buf_size);
                let light_fy = (1.0 - (light_ndc.y * 0.5 + 0.5)) * f32(params.light_buf_size);
                let light_px = i32(light_fx);
                let light_py = i32(light_fy);
                if light_px >= 0 && light_py >= 0 && u32(light_px) < params.light_buf_size && u32(light_py) < params.light_buf_size {
                    let light_idx = u32(light_py) * params.light_buf_size + u32(light_px);
                    let light_depth_enc = u32(clamp(1.0 - light_ndc.z, 0.0, 1.0) * 4294967295.0);
                    atomicMax(&points_light_depth[light_idx], light_depth_enc);
                }
            }
        }

        let clip = params.view_proj * world_pos4;
        if clip.w <= 0.0 { continue; }
        let ndc = clip.xyz / clip.w;
        if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 { continue; }

        let fx = (ndc.x * 0.5 + 0.5) * f32(params.ss_width);
        let fy = (1.0 - (ndc.y * 0.5 + 0.5)) * f32(params.ss_height);
        if fx < 0.0 || fy < 0.0 { continue; }

        rng ^= rng << 13u; rng ^= rng >> 17u; rng ^= rng << 5u;
        let noise_m = params.p[11].w;
        let fx_n = fx + (f32(rng >> 16u) / 32768.0 - 1.0) * noise_m;
        let fy_n = fy + (f32(rng & 0xFFFFu) / 32768.0 - 1.0) * noise_m;
        let px0 = i32(fx_n);
        let py0 = i32(fy_n);
        let ddx = fract(fx_n);
        let ddy = fract(fy_n);

        let w00 = u32((1.0 - ddx) * (1.0 - ddy) * WEIGHT_SCALE);
        let w10 = u32(       ddx  * (1.0 - ddy) * WEIGHT_SCALE);
        let w01 = u32((1.0 - ddx) *        ddy  * WEIGHT_SCALE);
        let w11 = 1024u - w00 - w10 - w01;

        let sc00 = speed_enc * w00 / SPEED_SCALE;
        let sc10 = speed_enc * w10 / SPEED_SCALE;
        let sc01 = speed_enc * w01 / SPEED_SCALE;
        let sc11 = speed_enc * w11 / SPEED_SCALE;

        let depth_enc = u32(clamp(1.0 - ndc.z, 0.0, 1.0) * 4294967295.0);
        let radius_i = i32(params.points_radius);
        for (var dy2 = -radius_i; dy2 <= radius_i; dy2++) {
            for (var dx2 = -radius_i; dx2 <= radius_i; dx2++) {
                let ppx = px0 + dx2;
                let ppy = py0 + dy2;
                if ppx >= 0 && ppy >= 0 && u32(ppx) < params.ss_width && u32(ppy) < params.ss_height {
                    let pidx = u32(ppy) * params.ss_width + u32(ppx);
                    atomicMax(&points_depth[pidx], depth_enc);
                    atomicAdd(&points_hit[pidx], 1u);
                }
            }
        }
        splat_pixel(px0,     py0,     w00, sc00);
        splat_pixel(px0 + 1, py0,     w10, sc10);
        splat_pixel(px0,     py0 + 1, w01, sc01);
        splat_pixel(px0 + 1, py0 + 1, w11, sc11);
    }

    states[traj] = vec4<f32>(x, y, z, bitcast<f32>(rng));
}
