mod commands;
mod dirty;
pub mod target;

use crate::backend::RenderTargetMode;
use crate::blend::ComplexBlend;
use crate::buffer_pool::TexturePool;
use crate::dynamic_transforms::DynamicTransforms;
use crate::filters::FilterSource;
use crate::mesh::Mesh;
use crate::pixel_bender::{ShaderMode, run_pixelbender_shader_impl};
use crate::surface::commands::{Chunk, CommandRenderer, chunk_blends};
use crate::utils::supported_sample_count;
use crate::{Descriptors, MaskState, Pipelines};
use ruffle_render::bitmap::PixelRegion;
use ruffle_render::commands::CommandList;
use ruffle_render::pixel_bender_support::{ImageInputTexture, PixelBenderShaderArgument};
use ruffle_render::quality::StageQuality;
use std::cell::RefCell;
use std::sync::Arc;
use target::{CommandTarget, create_region_bind_group};
use tracing::instrument;

use crate::utils::run_copy_pipeline;

pub use crate::surface::commands::LayerRef;

use self::commands::ChunkBlendMode;

#[derive(Debug)]
pub struct Surface {
    viewport: PixelRegion,
    size: wgpu::Extent3d,
    quality: StageQuality,
    sample_count: u32,
    pipelines: Arc<Pipelines>,
    format: wgpu::TextureFormat,
    dirty_tiles: RefCell<dirty::DirtyTileState>,
}

impl Surface {
    pub fn new(
        descriptors: &Descriptors,
        quality: StageQuality,
        width: u32,
        height: u32,
        frame_buffer_format: wgpu::TextureFormat,
    ) -> Self {
        Self::for_viewport(
            descriptors,
            quality,
            PixelRegion::for_whole_size(width, height),
            frame_buffer_format,
        )
    }

    pub fn for_viewport(
        descriptors: &Descriptors,
        quality: StageQuality,
        viewport: PixelRegion,
        frame_buffer_format: wgpu::TextureFormat,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: viewport.width(),
            height: viewport.height(),
            depth_or_array_layers: 1,
        };

        let sample_count = supported_sample_count(
            &descriptors.adapter,
            quality.sample_count(),
            frame_buffer_format,
        );
        let pipelines = descriptors.pipelines(sample_count, frame_buffer_format);
        Self {
            viewport,
            size,
            quality,
            sample_count,
            pipelines,
            format: frame_buffer_format,
            dirty_tiles: RefCell::new(dirty::DirtyTileState::new()),
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub fn draw_frame_commands_and_copy_to<'frame, 'global: 'frame>(
        &self,
        frame_view: &wgpu::TextureView,
        clear: wgpu::Color,
        descriptors: &'global Descriptors,
        staging_belt: &'frame mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'frame mut wgpu::CommandEncoder,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        texture_pool: &mut TexturePool,
    ) {
        if self.sample_count == 1 {
            return self.draw_commands_and_copy_to(
                frame_view,
                RenderTargetMode::FreshWithColor(clear),
                descriptors,
                staging_belt,
                dynamic_transforms,
                draw_encoder,
                meshes,
                commands,
                LayerRef::None,
                texture_pool,
            );
        }

        let decision = self.dirty_tiles.borrow_mut().prepare(
            descriptors,
            &commands,
            self.viewport,
            self.format,
            self.sample_count,
            clear,
        );
        if decision.rects.as_ref().is_some_and(Vec::is_empty) {
            let globals = texture_pool.get_globals(descriptors, self.viewport);
            let (_buffer, bind_group) = create_region_bind_group(descriptors, self.viewport);
            let view = decision.resolved.create_view(&Default::default());
            run_copy_pipeline(
                descriptors,
                self.format,
                frame_view,
                &view,
                &bind_group,
                &globals,
                1,
                draw_encoder,
            );
            return;
        }

        let partial = decision.rects.is_some();
        let target = self.draw_commands_impl(
            RenderTargetMode::RetainedMultisample {
                multisampled: decision.multisampled,
                resolved: decision.resolved,
                clear: (!partial).then_some(clear),
            },
            descriptors,
            meshes,
            commands,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            LayerRef::None,
            texture_pool,
            decision.rects.as_deref(),
            clear,
        );
        run_copy_pipeline(
            descriptors,
            self.format,
            frame_view,
            target.color_view(),
            target.whole_frame_bind_group(descriptors),
            target.globals(),
            1,
            draw_encoder,
        );
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands_and_copy_to<'frame, 'global: 'frame>(
        &self,
        frame_view: &wgpu::TextureView,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        staging_belt: &'frame mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'frame mut wgpu::CommandEncoder,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        layer: LayerRef,
        texture_pool: &mut TexturePool,
    ) {
        let target = self.draw_commands(
            render_target_mode,
            descriptors,
            meshes,
            commands,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            layer,
            texture_pool,
        );

        run_copy_pipeline(
            descriptors,
            self.format,
            frame_view,
            target.color_view(),
            target.whole_frame_bind_group(descriptors),
            target.globals(),
            1,
            draw_encoder,
        );
    }

    #[expect(clippy::too_many_arguments)]
    #[instrument(level = "debug", skip_all)]
    pub fn draw_commands<'frame, 'global: 'frame>(
        &self,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        staging_belt: &'global mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'frame mut wgpu::CommandEncoder,
        nearest_layer: LayerRef<'frame>,
        texture_pool: &mut TexturePool,
    ) -> CommandTarget {
        self.draw_commands_impl(
            render_target_mode,
            descriptors,
            meshes,
            commands,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            nearest_layer,
            texture_pool,
            None,
            wgpu::Color::TRANSPARENT,
        )
    }

    #[expect(clippy::too_many_arguments)]
    fn draw_commands_impl<'frame, 'global: 'frame>(
        &self,
        render_target_mode: RenderTargetMode,
        descriptors: &'global Descriptors,
        meshes: &'global Vec<Mesh>,
        commands: CommandList,
        staging_belt: &'global mut wgpu::util::StagingBelt,
        dynamic_transforms: &'global DynamicTransforms,
        draw_encoder: &'frame mut wgpu::CommandEncoder,
        nearest_layer: LayerRef<'frame>,
        texture_pool: &mut TexturePool,
        dirty_rects: Option<&[PixelRegion]>,
        clear: wgpu::Color,
    ) -> CommandTarget {
        let target = CommandTarget::for_viewport(
            descriptors,
            texture_pool,
            self.viewport,
            self.format,
            self.sample_count,
            render_target_mode,
            draw_encoder,
        );
        if let Some(rects) = dirty_rects {
            target.prepare_dirty_tiles(
                rects,
                clear,
                &self.pipelines,
                descriptors,
                texture_pool,
                draw_encoder,
            );
        }

        let mut num_masks = 0;
        let mut mask_state = MaskState::NoMask;
        let chunks = chunk_blends(
            commands,
            descriptors,
            staging_belt,
            dynamic_transforms,
            draw_encoder,
            meshes,
            self.quality,
            target.viewport(),
            dirty_rects,
            match nearest_layer {
                LayerRef::Current => LayerRef::Parent(&target),
                layer => layer,
            },
            texture_pool,
        );

        for chunk in chunks {
            match chunk {
                Chunk::Draw {
                    chunk,
                    needs_stencil,
                    transforms,
                } => {
                    transforms.copy_to(
                        staging_belt,
                        &descriptors.device,
                        draw_encoder,
                        &dynamic_transforms.buffer,
                    );
                    let dirty_tiles = dirty_rects.is_some();
                    let effective_needs_stencil = needs_stencil || dirty_tiles;
                    let mut render_pass =
                        draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: create_debug_label!(
                                "Chunked draw calls {}",
                                if effective_needs_stencil {
                                    "(with stencil)"
                                } else {
                                    "(Stencilless)"
                                }
                            )
                            .as_deref(),
                            color_attachments: &[target.color_attachments()],
                            depth_stencil_attachment: if effective_needs_stencil {
                                target.stencil_attachment(descriptors, texture_pool)
                            } else {
                                None
                            },
                            ..Default::default()
                        });
                    render_pass.set_bind_group(0, target.globals().bind_group(), &[]);
                    let mut renderer = CommandRenderer::new(
                        &self.pipelines,
                        descriptors,
                        dynamic_transforms,
                        render_pass,
                        num_masks,
                        mask_state,
                        effective_needs_stencil,
                        dirty_tiles,
                    );

                    for command in &chunk {
                        renderer.execute(command);
                    }

                    num_masks = renderer.num_masks();
                    mask_state = renderer.mask_state();
                }
                Chunk::Blend {
                    texture,
                    blend_mode: ChunkBlendMode::Shader(shader),
                    needs_stencil,
                    viewport,
                } => {
                    assert!(!needs_stencil, "Shader blend mode not implemented in masks");
                    let parent_blend_buffer = target.copy_region_to_blend_buffer(
                        viewport,
                        descriptors,
                        texture_pool,
                        draw_encoder,
                    );
                    run_pixelbender_shader_impl(
                        descriptors,
                        shader,
                        ShaderMode::Filter,
                        &[
                            PixelBenderShaderArgument::ImageInput {
                                index: 0,
                                channels: 0xFF,
                                name: "background".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(
                                    parent_blend_buffer.texture(),
                                )),
                            },
                            PixelBenderShaderArgument::ImageInput {
                                index: 1,
                                channels: 0xff,
                                name: "foreground".to_string(),
                                texture: Some(ImageInputTexture::TextureRef(texture.texture())),
                            },
                        ],
                        parent_blend_buffer.texture(),
                        draw_encoder,
                        target.color_attachments(),
                        target.sample_count(),
                        &FilterSource::for_entire_texture(texture.texture()),
                    )
                    .expect("Failed to run PixelBender blend mode");
                }
                Chunk::Blend {
                    texture,
                    blend_mode: ChunkBlendMode::Complex(blend_mode),
                    needs_stencil,
                    viewport,
                } => {
                    let dirty_tiles = dirty_rects.is_some();
                    let effective_needs_stencil = needs_stencil || dirty_tiles;
                    let parent = match blend_mode {
                        ComplexBlend::Alpha | ComplexBlend::Erase => {
                            match nearest_layer {
                                LayerRef::None => {
                                    // An Alpha or Erase with no Layer above it should be ignored
                                    continue;
                                }
                                LayerRef::Current => &target,
                                LayerRef::Parent(layer) => layer,
                            }
                        }
                        _ => &target,
                    };

                    let parent_blend_buffer = parent.copy_region_to_blend_buffer(
                        viewport,
                        descriptors,
                        texture_pool,
                        draw_encoder,
                    );

                    let blend_bind_group =
                        descriptors
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: create_debug_label!(
                                    "Complex blend binds {:?} {}",
                                    blend_mode,
                                    if effective_needs_stencil {
                                        "(with stencil)"
                                    } else {
                                        "(Stencilless)"
                                    }
                                )
                                .as_deref(),
                                layout: &descriptors.bind_layouts.blend,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            parent_blend_buffer.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::TextureView(
                                            texture.view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: wgpu::BindingResource::Sampler(
                                            descriptors.bitmap_samplers.get_sampler(false, false),
                                        ),
                                    },
                                ],
                            });

                    let (_region_buffer, region_bind_group) =
                        create_region_bind_group(descriptors, viewport);
                    let mut render_pass =
                        draw_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: create_debug_label!(
                                "Complex blend {:?} {}",
                                blend_mode,
                                if effective_needs_stencil {
                                    "(with stencil)"
                                } else {
                                    "(Stencilless)"
                                }
                            )
                            .as_deref(),
                            color_attachments: &[target.color_attachments()],
                            depth_stencil_attachment: if effective_needs_stencil {
                                target.stencil_attachment(descriptors, texture_pool)
                            } else {
                                None
                            },
                            ..Default::default()
                        });
                    render_pass.set_bind_group(0, target.globals().bind_group(), &[]);

                    if dirty_tiles {
                        let reference = match mask_state {
                            MaskState::NoMask => num_masks,
                            MaskState::DrawMaskStencil => num_masks - 1,
                            MaskState::DrawMaskedContent | MaskState::ClearMaskStencil => num_masks,
                        };
                        render_pass.set_stencil_reference(0x80 | reference);
                        render_pass.set_pipeline(
                            self.pipelines.complex_blends[blend_mode]
                                .dirty_pipeline_for(mask_state),
                        );
                    } else if needs_stencil {
                        match mask_state {
                            MaskState::NoMask => {}
                            MaskState::DrawMaskStencil => {
                                render_pass.set_stencil_reference(num_masks - 1);
                            }
                            MaskState::DrawMaskedContent => {
                                render_pass.set_stencil_reference(num_masks);
                            }
                            MaskState::ClearMaskStencil => {
                                render_pass.set_stencil_reference(num_masks);
                            }
                        }
                        render_pass.set_pipeline(
                            self.pipelines.complex_blends[blend_mode].pipeline_for(mask_state),
                        );
                    } else {
                        render_pass.set_pipeline(
                            self.pipelines.complex_blends[blend_mode].stencilless_pipeline(),
                        );
                    }

                    render_pass.set_bind_group(1, &region_bind_group, &[0]);
                    render_pass.set_bind_group(2, &blend_bind_group, &[]);

                    render_pass.set_vertex_buffer(0, descriptors.quad.vertices_pos.slice(..));
                    render_pass.set_index_buffer(
                        descriptors.quad.indices.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );

                    if let Some(rects) = dirty_rects {
                        let mut drew = false;
                        for dirty in rects {
                            let left = dirty.x_min.max(viewport.x_min);
                            let top = dirty.y_min.max(viewport.y_min);
                            let right = dirty.x_max.min(viewport.x_max);
                            let bottom = dirty.y_max.min(viewport.y_max);
                            if left < right && top < bottom {
                                render_pass
                                    .set_scissor_rect(left, top, right - left, bottom - top);
                                render_pass.draw_indexed(0..6, 0, 0..1);
                                drew = true;
                            }
                        }
                        debug_assert!(drew, "blend must intersect dirty rects");
                    } else {
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }
                }
            }
        }

        // If nothing happened, ensure it's cleared so we don't operate on garbage data
        target.ensure_cleared(draw_encoder);

        target
    }

    pub fn quality(&self) -> StageQuality {
        self.quality
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }
}
