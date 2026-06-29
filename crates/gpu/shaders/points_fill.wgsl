// Points mode bilateral gap-fill: runs once per frame, after dispatch_sim,
// over the raw camera-space depth/hit buffers (`points_depth`/`points_hit`,
// written by every attractor's compute shader -- see sim*.wgsl), producing
// filled copies for points_composite.wgsl to read. Texels that already have
// hits pass straight through unchanged; empty texels surrounded by valid
// data get a weighted average of their neighbors, weighted by *both* spatial
// distance and depth similarity (bilateral, not a plain Gaussian) so small
// single-point gaps get filled without smearing across a real depth
// discontinuity (e.g. a near strand crossing in front of a far one).
//
// Single-pass approximation: the first valid neighbor encountered in scan
// order becomes the "pivot" depth that subsequent neighbors are weighted
// against, rather than a true two-pass bilateral filter (which would need
// to know the nearest/most-representative neighbor's depth before weighting
// anything). Good enough for filling genuinely small gaps; not intended to
// bridge large holes.

struct FillParams {
    ss_width:  u32,
    ss_height: u32,
    _pad:      vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: FillParams;
@group(0) @binding(1) var<storage, read> points_depth: array<u32>;
@group(0) @binding(2) var<storage, read> points_hit: array<u32>;
@group(0) @binding(3) var<storage, read_write> points_depth_filled: array<u32>;
@group(0) @binding(4) var<storage, read_write> points_hit_filled: array<u32>;

const DEPTH_MAX: f32 = 4294967295.0;
const FILL_RADIUS: i32 = 4;
const SPATIAL_SIGMA: f32 = 2.0;
const DEPTH_SIGMA: f32 = 0.02; // in NDC-z units
const CONFIDENCE_THRESHOLD: f32 = 0.5;
// Texels with fewer than this many splats are treated as not-yet-trustworthy
// rather than solid surface -- a single stray hit (an unsettled trajectory
// during warmup, or a mid-simulation divergence-recovery jump) would
// otherwise show up as a permanent speck, since there's no density-based
// dilution here the way the histogram modes have. Below this count, a texel
// falls through to the neighbor search below instead of passing through
// directly, so it still gets reinforced/smoothed if real surface is nearby,
// or correctly drops to empty if it's genuinely isolated.
const MIN_HITS: u32 = 2u;

fn decode_depth(enc: u32) -> f32 {
    return 1.0 - f32(enc) / DEPTH_MAX;
}

fn encode_depth(ndc_z: f32) -> u32 {
    return u32(clamp(1.0 - ndc_z, 0.0, 1.0) * DEPTH_MAX);
}

@compute @workgroup_size(8, 8)
fn fill_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.ss_width || y >= params.ss_height {
        return;
    }
    let idx = y * params.ss_width + x;

    let hit = points_hit[idx];
    if hit >= MIN_HITS {
        points_depth_filled[idx] = points_depth[idx];
        points_hit_filled[idx] = hit;
        return;
    }

    var pivot = -1.0;
    var weight_sum = 0.0;
    var depth_sum = 0.0;
    var hit_sum = 0.0;

    for (var dy = -FILL_RADIUS; dy <= FILL_RADIUS; dy++) {
        for (var dx = -FILL_RADIUS; dx <= FILL_RADIUS; dx++) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            if nx < 0 || ny < 0 || u32(nx) >= params.ss_width || u32(ny) >= params.ss_height {
                continue;
            }
            let nidx = u32(ny) * params.ss_width + u32(nx);
            let nhit = points_hit[nidx];
            if nhit < MIN_HITS {
                continue;
            }
            let nd = decode_depth(points_depth[nidx]);
            if pivot < 0.0 {
                pivot = nd;
            }
            let spatial_d2 = f32(dx * dx + dy * dy);
            let depth_d = nd - pivot;
            let w = exp(-spatial_d2 / (2.0 * SPATIAL_SIGMA * SPATIAL_SIGMA))
                  * exp(-(depth_d * depth_d) / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));
            weight_sum += w;
            depth_sum += w * nd;
            hit_sum += w * f32(nhit);
        }
    }

    if weight_sum > CONFIDENCE_THRESHOLD {
        let avg_depth = depth_sum / weight_sum;
        points_depth_filled[idx] = encode_depth(avg_depth);
        points_hit_filled[idx] = max(u32(hit_sum / weight_sum), 1u);
    } else {
        points_depth_filled[idx] = 0u;
        points_hit_filled[idx] = 0u;
    }
}
