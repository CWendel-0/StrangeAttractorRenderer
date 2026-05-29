// Blit the Rgba16Float HDR intermediate texture onto the swapchain target.

@group(0) @binding(0) var hdr_tex:  texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;

struct VertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertOut;
    out.pos = vec4<f32>(pos[vi], 0.0, 1.0);
    // Y is flipped here (0.5 - y*0.5) because this shader uses textureSample with
    // UV origin at top-left, whereas de_h/composite use @builtin(position) with manual
    // pixel addressing (1.0 - uv.y) in the fragment shader.  Both approaches agree on
    // screen orientation; they differ only in which stage carries the flip.
    out.uv  = vec2<f32>(pos[vi].x * 0.5 + 0.5, 0.5 - pos[vi].y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    return textureSample(hdr_tex, hdr_samp, in.uv);
}
