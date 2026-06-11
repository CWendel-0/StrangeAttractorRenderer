// GPU-side Polynomial Sin discrete map.
//
// p[] flat layout (P0..P38):
//   p[0].xyzw=P0-P3   p[1].xyzw=P4-P7    p[2].xyzw=P8-P11
//   p[3].xyzw=P12-P15 p[4].xyzw=P16-P19  p[5].xyzw=P20-P23
//   p[6].xyzw=P24-P27 p[7].xyzw=P28-P31  p[8].xyzw=P32-P35
//   p[9].xyz =P36,P37,P38
//
// x' = P0  + P1·x  + P2·y  + P3·z  + P4·sin(P5·x+P6)   + P7·sin(P8·y+P9)   + P10·sin(P11·z+P12)
// y' = P13 + P14·x + P15·y + P16·z + P17·sin(P18·x+P19) + P20·sin(P21·y+P22) + P23·sin(P24·z+P25)
// z' = P26 + P27·x + P28·y + P29·z + P30·sin(P31·x+P32) + P33·sin(P34·y+P35) + P36·sin(P37·z+P38)

struct SimParams {
    view_proj: mat4x4<f32>,
    ss_width:  u32,
    ss_height: u32,
    steps:     u32,
    num_traj:  u32,
    p:         array<vec4<f32>, 12>,
}

@group(0) @binding(0) var<storage, read_write> states:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accum:       array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> max_vals: array<atomic<u32>, 2>;
@group(0) @binding(3) var<uniform>             params:      SimParams;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let traj = gid.x;
    if traj >= params.num_traj { return; }

    var x = states[traj].x;
    var y = states[traj].y;
    var z = states[traj].z;
    var rng = bitcast<u32>(states[traj].w);
    if rng == 0u { rng = traj + 1u; }

    let p0  = params.p[0].x; let p1  = params.p[0].y; let p2  = params.p[0].z; let p3  = params.p[0].w;
    let p4  = params.p[1].x; let p5  = params.p[1].y; let p6  = params.p[1].z; let p7  = params.p[1].w;
    let p8  = params.p[2].x; let p9  = params.p[2].y; let p10 = params.p[2].z; let p11 = params.p[2].w;
    let p12 = params.p[3].x; let p13 = params.p[3].y; let p14 = params.p[3].z; let p15 = params.p[3].w;
    let p16 = params.p[4].x; let p17 = params.p[4].y; let p18 = params.p[4].z; let p19 = params.p[4].w;
    let p20 = params.p[5].x; let p21 = params.p[5].y; let p22 = params.p[5].z; let p23 = params.p[5].w;
    let p24 = params.p[6].x; let p25 = params.p[6].y; let p26 = params.p[6].z; let p27 = params.p[6].w;
    let p28 = params.p[7].x; let p29 = params.p[7].y; let p30 = params.p[7].z; let p31 = params.p[7].w;
    let p32 = params.p[8].x; let p33 = params.p[8].y; let p34 = params.p[8].z; let p35 = params.p[8].w;
    let p36 = params.p[9].x; let p37 = params.p[9].y; let p38 = params.p[9].z;

    for (var i = 0u; i < params.steps; i++) {
        let nx = p0  + p1*x  + p2*y  + p3*z
               + p4  * sin(p5*x  + p6)
               + p7  * sin(p8*y  + p9)
               + p10 * sin(p11*z + p12);
        let ny = p13 + p14*x + p15*y + p16*z
               + p17 * sin(p18*x + p19)
               + p20 * sin(p21*y + p22)
               + p23 * sin(p24*z + p25);
        let nz = p26 + p27*x + p28*y + p29*z
               + p30 * sin(p31*x + p32)
               + p33 * sin(p34*y + p35)
               + p36 * sin(p37*z + p38);

        if abs(nx) > 1e4 || abs(ny) > 1e4 || abs(nz) > 1e4 {
            x = 0.1 + f32(traj % 32u) * 0.05;
            y = f32(traj % 16u) * 0.03;
            z = 0.0;
            for (var w = 0u; w < 200u; w++) {
                let wx = p0 + p1*x + p2*y + p3*z + p4*sin(p5*x+p6) + p7*sin(p8*y+p9) + p10*sin(p11*z+p12);
                let wy = p13 + p14*x + p15*y + p16*z + p17*sin(p18*x+p19) + p20*sin(p21*y+p22) + p23*sin(p24*z+p25);
                let wz = p26 + p27*x + p28*y + p29*z + p30*sin(p31*x+p32) + p33*sin(p34*y+p35) + p36*sin(p37*z+p38);
                if abs(wx) < 1e4 && abs(wy) < 1e4 && abs(wz) < 1e4 { x = wx; y = wy; z = wz; }
            }
            continue;
        }

        let speed_raw = distance(vec3<f32>(nx, ny, nz), vec3<f32>(x, y, z));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);
        atomicMax(&max_vals[1], speed_enc);
        x = nx; y = ny; z = nz;

        let clip = params.view_proj * vec4<f32>(x, y, z, 1.0);
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
        splat_pixel(px0,     py0,     w00, speed_enc * w00 / SPEED_SCALE);
        splat_pixel(px0 + 1, py0,     w10, speed_enc * w10 / SPEED_SCALE);
        splat_pixel(px0,     py0 + 1, w01, speed_enc * w01 / SPEED_SCALE);
        splat_pixel(px0 + 1, py0 + 1, w11, speed_enc * w11 / SPEED_SCALE);
    }

    states[traj] = vec4<f32>(x, y, z, bitcast<f32>(rng));
}
