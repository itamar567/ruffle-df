//use super::utils::create_debug_label;
use bytemuck::{Pod, Zeroable};
use ruffle_render::bitmap::PixelRegion;
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub struct Globals {
    bind_group: wgpu::BindGroup,
    _buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GlobalsUniform {
    view_matrix: [[f32; 4]; 4],
}

impl GlobalsUniform {
    fn for_viewport(viewport: PixelRegion) -> Self {
        let width = viewport.width() as f32;
        let height = viewport.height() as f32;
        Self {
            view_matrix: [
                [2.0 / width, 0.0, 0.0, 0.0],
                [0.0, -2.0 / height, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [
                    -1.0 - 2.0 * viewport.x_min as f32 / width,
                    1.0 + 2.0 * viewport.y_min as f32 / height,
                    0.0,
                    1.0,
                ],
            ],
        }
    }
}

impl Globals {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        viewport: PixelRegion,
    ) -> Self {
        let temp_label = create_debug_label!("Globals buffer");
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: temp_label.as_deref(),
            contents: bytemuck::cast_slice(&[GlobalsUniform::for_viewport(viewport)]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_label = create_debug_label!("Globals bind group");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: bind_group_label.as_deref(),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            bind_group,
            _buffer: buffer,
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[cfg(test)]
mod tests {
    use super::GlobalsUniform;
    use ruffle_render::bitmap::PixelRegion;

    #[test]
    fn view_matrix_offsets_coordinates_to_viewport_origin() {
        let uniform = GlobalsUniform::for_viewport(PixelRegion::for_region(100, 50, 400, 200));

        assert_eq!(uniform.view_matrix[0][0], 0.005);
        assert_eq!(uniform.view_matrix[1][1], -0.01);
        assert_eq!(uniform.view_matrix[3], [-1.5, 1.5, 0.0, 1.0]);
    }
}
