// Draws a screen-aligned quad for dirty-tile stencil marking and exact background clears.
// `add_color` is passed through without premultiplication so it matches a render-pass clear.

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> transforms: common__Transforms;

@vertex
fn main_vertex(in: VertexInput) -> VertexOutput {
    let position = common__globals.view_matrix * transforms.world_matrix
        * vec4<f32>(in.position, 0.0, 1.0);
    return VertexOutput(position, transforms.add_color);
}

@fragment
fn main_fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
