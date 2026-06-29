// GPU-side Yang-Cao map. The true dynamical state is the coupled pair
// (w, s); the splatted point (x, y) is a derived projection recomputed
// from the freshly-updated (w, s) every iteration.
//
// Params in p[]:
//   p[0].x = a,  p[0].y = b,  p[0].z = c,  p[0].w = u
//   p[1].x = m
//
// Equations:
//   t  = c - 6/(1 + w^2 + s^2)
//   w' = 1 + u*(w*cos(t) - s*sin(t))
//   s' = u*(w*sin(t) + s*cos(t))
//   x  = 1 - a*s'^2 + b*w'
//   y  = m*w'*(1 - s')
//   plotted state: (x, y); stored state: (w, s)

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

// Returns (w', s', x, y): the advanced (w, s) state and the derived plot
// point (x, y) computed from that freshly-updated state.
fn map_step(w: f32, s: f32, a: f32, b: f32, c: f32, u: f32, m: f32) -> vec4<f32> {
    let t = c - 6.0 / (1.0 + w * w + s * s);
    let w2 = 1.0 + u * (w * cos(t) - s * sin(t));
    let s2 = u * (w * sin(t) + s * cos(t));
    let x = 1.0 - a * s2 * s2 + b * w2;
    let y = m * w2 * (1.0 - s2);
    return vec4<f32>(w2, s2, x, y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let traj = gid.x;
    if traj >= params.num_traj { return; }

    var w = states[traj].x;
    var s = states[traj].y;
    var rng = bitcast<u32>(states[traj].w);
    if rng == 0u { rng = traj + 1u; }

    let a = params.p[0].x;
    let b = params.p[0].y;
    let c = params.p[0].z;
    let u = params.p[0].w;
    let m = params.p[1].x;

    // Speed is measured between consecutive plot points produced within
    // this dispatch; the very first iteration of a dispatch has no prior
    // plot point to compare against, so it is skipped (negligible one-step
    // warm-start cost, same as other map shaders pay on dispatch boundaries).
    var prev_plot = vec2<f32>(0.0, 0.0);
    var have_prev = false;

    for (var i = 0u; i < params.steps; i++) {
        let r = map_step(w, s, a, b, c, u, m);
        let nw = r.x;
        let ns = r.y;
        let plot = r.zw;

        if abs(nw) > 1e6 || abs(ns) > 1e6 || plot.x != plot.x || plot.y != plot.y {
            w = recovery_hash(traj) * 0.2 + 0.05;
            s = recovery_hash(traj ^ 0x9E3779B9u) * 0.2 + 0.05;
            for (var ww = 0u; ww < 200u; ww++) {
                let rw = map_step(w, s, a, b, c, u, m);
                w = rw.x;
                s = rw.y;
            }
            have_prev = false;
            continue;
        }

        w = nw;
        s = ns;

        if have_prev {
            let speed_raw = distance(plot, prev_plot);
            let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);
            atomicMax(&max_vals[1], speed_enc);

            let world_pos4 = vec4<f32>(plot.x, plot.y, 0.0, 1.0);

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
            if clip.w > 0.0 {
                let ndc = clip.xyz / clip.w;
                if ndc.x >= -1.0 && ndc.x <= 1.0 && ndc.y >= -1.0 && ndc.y <= 1.0 {
                    let fx = (ndc.x * 0.5 + 0.5) * f32(params.ss_width);
                    let fy = (1.0 - (ndc.y * 0.5 + 0.5)) * f32(params.ss_height);
                    if fx >= 0.0 && fy >= 0.0 {
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
                }
            }
        }

        prev_plot = plot;
        have_prev = true;
    }

    states[traj] = vec4<f32>(w, s, 0.0, bitcast<f32>(rng));
}
