// GPU-side Polynomial Sprott (Sprott 3D quadratic/cubic/quartic/quintic map).
//
// p[] flat layout (169 floats = 43 vec4s):
//   p[0].x      = P0  = order (2.0 / 3.0 / 4.0 / 5.0)
//   p[0].y..p[14].x   = P1..P56   = X equation coefficients (cx[0..55])
//   p[14].y..p[28].x  = P57..P112 = Y equation coefficients (cy[0..55])
//   p[28].y..p[42].x  = P113..P168 = Z equation coefficients (cz[0..55])
//
// Monomial ordering matches Sprott's book (Chaoscope-compatible):
//   [0]=1   [1]=x   [2]=x²  [3]=xy  [4]=xz  [5]=y   [6]=y²  [7]=yz  [8]=z   [9]=z²
//   [10]=x³ [11]=x²y [12]=x²z [13]=xy² [14]=xyz [15]=xz² [16]=y³ [17]=y²z [18]=yz² [19]=z³
//   [20]=x⁴ [21]=x³y [22]=x³z [23]=x²y² [24]=x²yz [25]=x²z² [26]=xy³ [27]=xy²z
//   [28]=xyz² [29]=xz³ [30]=y⁴ [31]=y³z [32]=y²z² [33]=yz³ [34]=z⁴
//   [35]=x⁵ [36]=x⁴y [37]=x⁴z [38]=x³y² [39]=x³yz [40]=x³z² [41]=x²y³ [42]=x²y²z
//   [43]=x²yz² [44]=x²z³ [45]=xy⁴ [46]=xy³z [47]=xy²z² [48]=xyz³ [49]=xz⁴
//   [50]=y⁵ [51]=y⁴z [52]=y³z² [53]=y²z³ [54]=yz⁴ [55]=z⁵

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
    p:         array<vec4<f32>, 43>,
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
    let upx = u32(px); let upy = u32(py);
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

// Deterministic per-trajectory hash used to seed divergence recovery so every
// trajectory recovers to a distinct point. Without this, trajectories that
// diverge and share a small modulus would warm up to bit-identical states and
// splat duplicate points into the histogram for the rest of the render.
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

    let order = u32(params.p[0].x);

    // ---- X equation coefficients (P1-P56) ----
    let cx0  = params.p[0].y;  let cx1  = params.p[0].z;  let cx2  = params.p[0].w;
    let cx3  = params.p[1].x;  let cx4  = params.p[1].y;  let cx5  = params.p[1].z;  let cx6  = params.p[1].w;
    let cx7  = params.p[2].x;  let cx8  = params.p[2].y;  let cx9  = params.p[2].z;  let cx10 = params.p[2].w;
    let cx11 = params.p[3].x;  let cx12 = params.p[3].y;  let cx13 = params.p[3].z;  let cx14 = params.p[3].w;
    let cx15 = params.p[4].x;  let cx16 = params.p[4].y;  let cx17 = params.p[4].z;  let cx18 = params.p[4].w;
    let cx19 = params.p[5].x;  let cx20 = params.p[5].y;  let cx21 = params.p[5].z;  let cx22 = params.p[5].w;
    let cx23 = params.p[6].x;  let cx24 = params.p[6].y;  let cx25 = params.p[6].z;  let cx26 = params.p[6].w;
    let cx27 = params.p[7].x;  let cx28 = params.p[7].y;  let cx29 = params.p[7].z;  let cx30 = params.p[7].w;
    let cx31 = params.p[8].x;  let cx32 = params.p[8].y;  let cx33 = params.p[8].z;  let cx34 = params.p[8].w;
    let cx35 = params.p[9].x;  let cx36 = params.p[9].y;  let cx37 = params.p[9].z;  let cx38 = params.p[9].w;
    let cx39 = params.p[10].x; let cx40 = params.p[10].y; let cx41 = params.p[10].z; let cx42 = params.p[10].w;
    let cx43 = params.p[11].x; let cx44 = params.p[11].y; let cx45 = params.p[11].z; let cx46 = params.p[11].w;
    let cx47 = params.p[12].x; let cx48 = params.p[12].y; let cx49 = params.p[12].z; let cx50 = params.p[12].w;
    let cx51 = params.p[13].x; let cx52 = params.p[13].y; let cx53 = params.p[13].z; let cx54 = params.p[13].w;
    let cx55 = params.p[14].x;

    // ---- Y equation coefficients (P57-P112) ----
    let cy0  = params.p[14].y; let cy1  = params.p[14].z; let cy2  = params.p[14].w;
    let cy3  = params.p[15].x; let cy4  = params.p[15].y; let cy5  = params.p[15].z; let cy6  = params.p[15].w;
    let cy7  = params.p[16].x; let cy8  = params.p[16].y; let cy9  = params.p[16].z; let cy10 = params.p[16].w;
    let cy11 = params.p[17].x; let cy12 = params.p[17].y; let cy13 = params.p[17].z; let cy14 = params.p[17].w;
    let cy15 = params.p[18].x; let cy16 = params.p[18].y; let cy17 = params.p[18].z; let cy18 = params.p[18].w;
    let cy19 = params.p[19].x; let cy20 = params.p[19].y; let cy21 = params.p[19].z; let cy22 = params.p[19].w;
    let cy23 = params.p[20].x; let cy24 = params.p[20].y; let cy25 = params.p[20].z; let cy26 = params.p[20].w;
    let cy27 = params.p[21].x; let cy28 = params.p[21].y; let cy29 = params.p[21].z; let cy30 = params.p[21].w;
    let cy31 = params.p[22].x; let cy32 = params.p[22].y; let cy33 = params.p[22].z; let cy34 = params.p[22].w;
    let cy35 = params.p[23].x; let cy36 = params.p[23].y; let cy37 = params.p[23].z; let cy38 = params.p[23].w;
    let cy39 = params.p[24].x; let cy40 = params.p[24].y; let cy41 = params.p[24].z; let cy42 = params.p[24].w;
    let cy43 = params.p[25].x; let cy44 = params.p[25].y; let cy45 = params.p[25].z; let cy46 = params.p[25].w;
    let cy47 = params.p[26].x; let cy48 = params.p[26].y; let cy49 = params.p[26].z; let cy50 = params.p[26].w;
    let cy51 = params.p[27].x; let cy52 = params.p[27].y; let cy53 = params.p[27].z; let cy54 = params.p[27].w;
    let cy55 = params.p[28].x;

    // ---- Z equation coefficients (P113-P168) ----
    let cz0  = params.p[28].y; let cz1  = params.p[28].z; let cz2  = params.p[28].w;
    let cz3  = params.p[29].x; let cz4  = params.p[29].y; let cz5  = params.p[29].z; let cz6  = params.p[29].w;
    let cz7  = params.p[30].x; let cz8  = params.p[30].y; let cz9  = params.p[30].z; let cz10 = params.p[30].w;
    let cz11 = params.p[31].x; let cz12 = params.p[31].y; let cz13 = params.p[31].z; let cz14 = params.p[31].w;
    let cz15 = params.p[32].x; let cz16 = params.p[32].y; let cz17 = params.p[32].z; let cz18 = params.p[32].w;
    let cz19 = params.p[33].x; let cz20 = params.p[33].y; let cz21 = params.p[33].z; let cz22 = params.p[33].w;
    let cz23 = params.p[34].x; let cz24 = params.p[34].y; let cz25 = params.p[34].z; let cz26 = params.p[34].w;
    let cz27 = params.p[35].x; let cz28 = params.p[35].y; let cz29 = params.p[35].z; let cz30 = params.p[35].w;
    let cz31 = params.p[36].x; let cz32 = params.p[36].y; let cz33 = params.p[36].z; let cz34 = params.p[36].w;
    let cz35 = params.p[37].x; let cz36 = params.p[37].y; let cz37 = params.p[37].z; let cz38 = params.p[37].w;
    let cz39 = params.p[38].x; let cz40 = params.p[38].y; let cz41 = params.p[38].z; let cz42 = params.p[38].w;
    let cz43 = params.p[39].x; let cz44 = params.p[39].y; let cz45 = params.p[39].z; let cz46 = params.p[39].w;
    let cz47 = params.p[40].x; let cz48 = params.p[40].y; let cz49 = params.p[40].z; let cz50 = params.p[40].w;
    let cz51 = params.p[41].x; let cz52 = params.p[41].y; let cz53 = params.p[41].z; let cz54 = params.p[41].w;
    let cz55 = params.p[42].x;

    for (var i = 0u; i < params.steps; i++) {
        // Precompute all powers (unused ones for lower orders are eliminated by the
        // compiler since their coefficients are 0.0 in the uniform).
        let x2 = x*x; let x3 = x2*x; let x4 = x3*x; let x5 = x4*x;
        let y2 = y*y; let y3 = y2*y; let y4 = y3*y; let y5 = y4*y;
        let z2 = z*z; let z3 = z2*z; let z4 = z3*z; let z5 = z4*z;

        // Degree ≤ 2 (always) — Sprott ordering: 1, x, x², xy, xz, y, y², yz, z, z²
        var nx = cx0 + cx1*x + cx2*x2 + cx3*(x*y) + cx4*(x*z)
               + cx5*y + cx6*y2 + cx7*(y*z) + cx8*z + cx9*z2;
        var ny = cy0 + cy1*x + cy2*x2 + cy3*(x*y) + cy4*(x*z)
               + cy5*y + cy6*y2 + cy7*(y*z) + cy8*z + cy9*z2;
        var nz = cz0 + cz1*x + cz2*x2 + cz3*(x*y) + cz4*(x*z)
               + cz5*y + cz6*y2 + cz7*(y*z) + cz8*z + cz9*z2;

        // Degree 3
        if order >= 3u {
            nx += cx10*x3 + cx11*(x2*y) + cx12*(x2*z) + cx13*(x*y2)
                + cx14*(x*y*z) + cx15*(x*z2) + cx16*y3 + cx17*(y2*z)
                + cx18*(y*z2) + cx19*z3;
            ny += cy10*x3 + cy11*(x2*y) + cy12*(x2*z) + cy13*(x*y2)
                + cy14*(x*y*z) + cy15*(x*z2) + cy16*y3 + cy17*(y2*z)
                + cy18*(y*z2) + cy19*z3;
            nz += cz10*x3 + cz11*(x2*y) + cz12*(x2*z) + cz13*(x*y2)
                + cz14*(x*y*z) + cz15*(x*z2) + cz16*y3 + cz17*(y2*z)
                + cz18*(y*z2) + cz19*z3;
        }

        // Degree 4
        if order >= 4u {
            nx += cx20*x4 + cx21*(x3*y) + cx22*(x3*z) + cx23*(x2*y2)
                + cx24*(x2*y*z) + cx25*(x2*z2) + cx26*(x*y3) + cx27*(x*y2*z)
                + cx28*(x*y*z2) + cx29*(x*z3) + cx30*y4 + cx31*(y3*z)
                + cx32*(y2*z2) + cx33*(y*z3) + cx34*z4;
            ny += cy20*x4 + cy21*(x3*y) + cy22*(x3*z) + cy23*(x2*y2)
                + cy24*(x2*y*z) + cy25*(x2*z2) + cy26*(x*y3) + cy27*(x*y2*z)
                + cy28*(x*y*z2) + cy29*(x*z3) + cy30*y4 + cy31*(y3*z)
                + cy32*(y2*z2) + cy33*(y*z3) + cy34*z4;
            nz += cz20*x4 + cz21*(x3*y) + cz22*(x3*z) + cz23*(x2*y2)
                + cz24*(x2*y*z) + cz25*(x2*z2) + cz26*(x*y3) + cz27*(x*y2*z)
                + cz28*(x*y*z2) + cz29*(x*z3) + cz30*y4 + cz31*(y3*z)
                + cz32*(y2*z2) + cz33*(y*z3) + cz34*z4;
        }

        // Degree 5
        if order >= 5u {
            nx += cx35*x5  + cx36*(x4*y)   + cx37*(x4*z)   + cx38*(x3*y2)
                + cx39*(x3*y*z) + cx40*(x3*z2) + cx41*(x2*y3) + cx42*(x2*y2*z)
                + cx43*(x2*y*z2) + cx44*(x2*z3) + cx45*(x*y4) + cx46*(x*y3*z)
                + cx47*(x*y2*z2) + cx48*(x*y*z3) + cx49*(x*z4) + cx50*y5
                + cx51*(y4*z) + cx52*(y3*z2) + cx53*(y2*z3) + cx54*(y*z4) + cx55*z5;
            ny += cy35*x5  + cy36*(x4*y)   + cy37*(x4*z)   + cy38*(x3*y2)
                + cy39*(x3*y*z) + cy40*(x3*z2) + cy41*(x2*y3) + cy42*(x2*y2*z)
                + cy43*(x2*y*z2) + cy44*(x2*z3) + cy45*(x*y4) + cy46*(x*y3*z)
                + cy47*(x*y2*z2) + cy48*(x*y*z3) + cy49*(x*z4) + cy50*y5
                + cy51*(y4*z) + cy52*(y3*z2) + cy53*(y2*z3) + cy54*(y*z4) + cy55*z5;
            nz += cz35*x5  + cz36*(x4*y)   + cz37*(x4*z)   + cz38*(x3*y2)
                + cz39*(x3*y*z) + cz40*(x3*z2) + cz41*(x2*y3) + cz42*(x2*y2*z)
                + cz43*(x2*y*z2) + cz44*(x2*z3) + cz45*(x*y4) + cz46*(x*y3*z)
                + cz47*(x*y2*z2) + cz48*(x*y*z3) + cz49*(x*z4) + cz50*y5
                + cz51*(y4*z) + cz52*(y3*z2) + cz53*(y2*z3) + cz54*(y*z4) + cz55*z5;
        }

        if abs(nx) > 1e4 || abs(ny) > 1e4 || abs(nz) > 1e4 {
            // Every coordinate is derived from a differently-salted hash of the
            // full trajectory index, so the recovered state is unique per
            // trajectory (see recovery_hash above).
            x = 0.1 + recovery_hash(traj) * 1.6;
            y = recovery_hash(traj ^ 0x9E3779B9u) * 0.48;
            z = (recovery_hash(traj ^ 0x85EBCA6Bu) - 0.5) * 0.1;
            // Warm-up using degree ≤ 2 terms only (sufficient for convergence near origin)
            for (var w = 0u; w < 200u; w++) {
                let wx2 = x*x; let wy2 = y*y; let wz2 = z*z;
                let wx = cx0 + cx1*x + cx2*wx2 + cx3*(x*y) + cx4*(x*z) + cx5*y + cx6*wy2 + cx7*(y*z) + cx8*z + cx9*wz2;
                let wy = cy0 + cy1*x + cy2*wx2 + cy3*(x*y) + cy4*(x*z) + cy5*y + cy6*wy2 + cy7*(y*z) + cy8*z + cy9*wz2;
                let wz = cz0 + cz1*x + cz2*wx2 + cz3*(x*y) + cz4*(x*z) + cz5*y + cz6*wy2 + cz7*(y*z) + cz8*z + cz9*wz2;
                if abs(wx) < 1e4 && abs(wy) < 1e4 && abs(wz) < 1e4 { x = wx; y = wy; z = wz; }
            }
            continue;
        }

        let speed_raw = distance(vec3<f32>(nx, ny, nz), vec3<f32>(x, y, z));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);
        atomicMax(&max_vals[1], speed_enc);
        x = nx; y = ny; z = nz;

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
        let px0 = i32(fx_n); let py0 = i32(fy_n);
        let ddx = fract(fx_n); let ddy = fract(fy_n);
        let w00 = u32((1.0 - ddx) * (1.0 - ddy) * WEIGHT_SCALE);
        let w10 = u32(       ddx  * (1.0 - ddy) * WEIGHT_SCALE);
        let w01 = u32((1.0 - ddx) *        ddy  * WEIGHT_SCALE);
        let w11 = 1024u - w00 - w10 - w01;
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
        splat_pixel(px0,     py0,     w00, speed_enc * w00 / SPEED_SCALE);
        splat_pixel(px0 + 1, py0,     w10, speed_enc * w10 / SPEED_SCALE);
        splat_pixel(px0,     py0 + 1, w01, speed_enc * w01 / SPEED_SCALE);
        splat_pixel(px0 + 1, py0 + 1, w11, speed_enc * w11 / SPEED_SCALE);
    }

    states[traj] = vec4<f32>(x, y, z, bitcast<f32>(rng));
}
