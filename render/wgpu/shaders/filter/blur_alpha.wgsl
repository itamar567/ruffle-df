// NOTE: The `shader_filter_common.wgsl` source is prepended to this before compilation.

struct Filter {
    // Secretly a vec2<f32> but within alignment rules.
    dir_x: f32,
    dir_y: f32,

    full_size: f32,
    m: f32,
    m2: f32,
    first_weight: f32,
    last_offset: f32,
    last_weight: f32,
}

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var<uniform> filter_args: Filter;

@vertex
fn main_vertex(in: filter__VertexInput) -> filter__VertexOutput {
    var result = filter__main_vertex(in);

    let direction = vec2<f32>(filter_args.dir_x, filter_args.dir_y);
    result.uv -= direction * filter_args.m;

    return result;
}

fn blur_alpha(in: filter__VertexOutput, source_is_red: bool) -> f32 {
    let direction = vec2<f32>(filter_args.dir_x, filter_args.dir_y);

    let first = textureSample(texture, texture_sampler, in.uv - direction);
    var total = select(first.a, first.r, source_is_red) * filter_args.first_weight;

    var center = 0.0;
    for (var i = 0.5; i < filter_args.m2; i += 2.0) {
        let sample = textureSample(texture, texture_sampler, in.uv + direction * i);
        center += select(sample.a, sample.r, source_is_red);
    }
    total += center * 2.0;

    let last_location = in.uv + direction * (filter_args.m2 + filter_args.last_offset);
    let last = textureSample(texture, texture_sampler, last_location);
    total += select(last.a, last.r, source_is_red) * filter_args.last_weight;

    let result = total / filter_args.full_size;
    return floor(result * 255.0) / 255.0;
}

@fragment
fn main_fragment_from_alpha(in: filter__VertexOutput) -> @location(0) f32 {
    return blur_alpha(in, false);
}

@fragment
fn main_fragment_from_red(in: filter__VertexOutput) -> @location(0) f32 {
    return blur_alpha(in, true);
}
