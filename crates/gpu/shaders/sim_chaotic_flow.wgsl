// GPU-side Chaotic Flow attractor.
//
// Param layout in p[] (flat index → vec4 element):
//   p[0].xyzw = m0,  m1,  m2,  m3
//   p[1].xyzw = m4,  m5,  m6,  m7
//   p[2].xyzw = m8,  m9,  m10, m11
//   p[3].xyz  = Op0, Op1, Op2          (ops for X eq: m0,m1,m2)
//   p[3].w    = Op4                    (op  for Y eq: m4)
//   p[4].xy   = Op5, Op6               (ops for Y eq: m5,m6)
//   p[4].zw   = Op8, Op9               (ops for Z eq: m8,m9)
//   p[5].x    = Op10                   (op  for Z eq: m10)
//   p[5].y    = dt
//
// m3, m7, m11 are always constant terms (no Op selector), matching Chaoscope.
//
// Equations:
//   dx = m0·v(Op0) + m1·v(Op1) + m2·v(Op2) + m3
//   dy = m4·v(Op4) + m5·v(Op5) + m6·v(Op6) + m7
//   dz = m8·v(Op8) + m9·v(Op9) + m10·v(Op10) + m11
//   x' = x + dt·dx   (Euler)
//   where v(0/"-")=1, v(1/"X")=x, v(2/"Y")=y, v(3/"Z")=z

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
@group(0) @binding(2) var<storage, read_write> max_density: atomic<u32>;
@group(0) @binding(3) var<uniform>             params:      SimParams;

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
        atomicMax(&max_density, prev + weight);
    }
    if atomicLoad(&accum[base + 1u]) < 0x7FFFFFFFu {
        atomicAdd(&accum[base + 1u], speed_contrib);
    }
}

fn eval_v(op: u32, x: f32, y: f32, z: f32) -> f32 {
    if op == 1u { return x; }
    if op == 2u { return y; }
    if op == 3u { return z; }
    return 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let traj = gid.x;
    if traj >= params.num_traj { return; }

    var x = states[traj].x;
    var y = states[traj].y;
    var z = states[traj].z;

    let m0  = params.p[0].x; let m1  = params.p[0].y; let m2  = params.p[0].z; let m3  = params.p[0].w;
    let m4  = params.p[1].x; let m5  = params.p[1].y; let m6  = params.p[1].z; let m7  = params.p[1].w;
    let m8  = params.p[2].x; let m9  = params.p[2].y; let m10 = params.p[2].z; let m11 = params.p[2].w;

    let op0  = u32(params.p[3].x);
    let op1  = u32(params.p[3].y);
    let op2  = u32(params.p[3].z);
    let op4  = u32(params.p[3].w);
    let op5  = u32(params.p[4].x);
    let op6  = u32(params.p[4].y);
    let op8  = u32(params.p[4].z);
    let op9  = u32(params.p[4].w);
    let op10 = u32(params.p[5].x);
    let dt   = params.p[5].y;

    for (var i = 0u; i < params.steps; i++) {
        let vx = m0*eval_v(op0,x,y,z)*x + m1*eval_v(op1,x,y,z)*y + m2*eval_v(op2,x,y,z)*z + m3;
        let vy = m4*eval_v(op4,x,y,z)*x + m5*eval_v(op5,x,y,z)*y + m6*eval_v(op6,x,y,z)*z + m7;
        let vz = m8*eval_v(op8,x,y,z)*x + m9*eval_v(op9,x,y,z)*y + m10*eval_v(op10,x,y,z)*z + m11;

        let nx = x + dt * vx;
        let ny = y + dt * vy;
        let nz = z + dt * vz;
        x = nx; y = ny; z = nz;

        if abs(x) > 1e4 || abs(y) > 1e4 || abs(z) > 1e4 {
            x = 0.0; y = 0.1 + f32(traj % 32u) * 0.02; z = 0.0;
            for (var w = 0u; w < 500u; w++) {
                let wx = x; let wy = y; let wz = z;
                let wvx = m0*eval_v(op0,wx,wy,wz)*wx + m1*eval_v(op1,wx,wy,wz)*wy
                        + m2*eval_v(op2,wx,wy,wz)*wz + m3;
                let wvy = m4*eval_v(op4,wx,wy,wz)*wx + m5*eval_v(op5,wx,wy,wz)*wy
                        + m6*eval_v(op6,wx,wy,wz)*wz + m7;
                let wvz = m8*eval_v(op8,wx,wy,wz)*wx + m9*eval_v(op9,wx,wy,wz)*wy
                        + m10*eval_v(op10,wx,wy,wz)*wz + m11;
                x = wx + dt * wvx; y = wy + dt * wvy; z = wz + dt * wvz;
            }
            continue;
        }

        let speed_raw = length(vec3<f32>(vx, vy, vz));
        let speed_enc = min(u32(log(speed_raw + 1.0) * 32.0), 255u);

        let clip = params.view_proj * vec4<f32>(x, y, z, 1.0);
        if clip.w <= 0.0 { continue; }
        let ndc = clip.xyz / clip.w;
        if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 { continue; }

        let fx = (ndc.x * 0.5 + 0.5) * f32(params.ss_width);
        let fy = (1.0 - (ndc.y * 0.5 + 0.5)) * f32(params.ss_height);
        if fx < 0.0 || fy < 0.0 { continue; }

        let px0 = i32(fx);
        let py0 = i32(fy);
        let ddx = fract(fx);
        let ddy = fract(fy);

        let w00 = u32((1.0 - ddx) * (1.0 - ddy) * WEIGHT_SCALE);
        let w10 = u32(       ddx  * (1.0 - ddy) * WEIGHT_SCALE);
        let w01 = u32((1.0 - ddx) *        ddy  * WEIGHT_SCALE);
        let w11 = 1024u - w00 - w10 - w01;

        let sc00 = speed_enc * w00 / SPEED_SCALE;
        let sc10 = speed_enc * w10 / SPEED_SCALE;
        let sc01 = speed_enc * w01 / SPEED_SCALE;
        let sc11 = speed_enc * w11 / SPEED_SCALE;

        splat_pixel(px0,     py0,     w00, sc00);
        splat_pixel(px0 + 1, py0,     w10, sc10);
        splat_pixel(px0,     py0 + 1, w01, sc01);
        splat_pixel(px0 + 1, py0 + 1, w11, sc11);
    }

    states[traj] = vec4<f32>(x, y, z, 0.0);
}
