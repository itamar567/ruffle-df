use crate::Transforms;
use crate::descriptors::Descriptors;
use std::mem;

/// Number of transforms a single dynamic uniform buffer holds before the
/// renderer must split the draw stream into a new render pass. Sized for the
/// worst common case (full redraw frames submit thousands of draws; each split
/// costs a full-surface multisampled load/store/resolve).
const TRANSFORMS_PER_BUFFER: u64 = 1024;

pub struct DynamicTransforms {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

/// Byte size of the dynamic uniform buffer that holds up to
/// `TRANSFORMS_PER_BUFFER` aligned transform entries, capped by the device's
/// maximum buffer size.
fn transforms_buffer_size(limits: &wgpu::Limits) -> u64 {
    let stride = mem::size_of::<Transforms>() as u64;
    let alignment = limits.min_uniform_buffer_offset_alignment as u64;
    let aligned_stride = if alignment > 0 {
        stride.div_ceil(alignment) * alignment
    } else {
        stride
    };
    (aligned_stride * TRANSFORMS_PER_BUFFER).min(limits.max_buffer_size)
}

impl DynamicTransforms {
    pub fn new(descriptors: &Descriptors) -> Self {
        let buffer = descriptors.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: transforms_buffer_size(&descriptors.limits),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = descriptors
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &descriptors.bind_layouts.transforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(mem::size_of::<Transforms>() as u64),
                    }),
                }],
            });
        Self { buffer, bind_group }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(alignment: u32, max_buffer_size: u64) -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        limits.min_uniform_buffer_offset_alignment = alignment;
        limits.max_buffer_size = max_buffer_size;
        limits
    }

    #[test]
    fn buffer_holds_one_thousand_transforms_at_gl_alignment() {
        let size = transforms_buffer_size(&limits(256, u64::MAX));
        assert_eq!(size, 1024 * 256);
    }

    #[test]
    fn buffer_falls_back_to_struct_size_without_alignment() {
        let size = transforms_buffer_size(&limits(0, u64::MAX));
        assert_eq!(size, 1024 * mem::size_of::<Transforms>() as u64);
    }

    #[test]
    fn buffer_is_capped_by_max_buffer_size() {
        let size = transforms_buffer_size(&limits(256, 512 * 256));
        assert_eq!(size, 512 * 256);
    }
}
