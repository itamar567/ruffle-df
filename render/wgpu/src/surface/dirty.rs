use crate::blend::{BlendType, ComplexBlend, TrivialBlend};
use crate::mesh::{DrawType, as_mesh};
use crate::{Descriptors, as_texture};
use ruffle_render::bitmap::{PixelRegion, PixelSnapping};
use ruffle_render::commands::{Command, CommandList};
use ruffle_render::matrix::Matrix;
use std::sync::Arc;
use swf::ColorTransform;

pub const TILE_SIZE: u32 = 32;
const MAX_DIRTY_TILE_FRACTION_DENOMINATOR: usize = 4;
const HIGH_CHANGE_STREAK_LIMIT: u8 = 3;
const TRACKING_COOLDOWN_FRAMES: u8 = 10;

#[derive(Debug)]
struct RetainedFrame {
    multisampled: wgpu::Texture,
    resolved: wgpu::Texture,
}

#[derive(Debug)]
pub struct DirtyTileState {
    previous: Vec<u64>,
    retained: Option<RetainedFrame>,
    high_change_streak: u8,
    tracking_cooldown: u8,
}

#[derive(Debug)]
pub struct DirtyDecision {
    pub multisampled: wgpu::Texture,
    pub resolved: wgpu::Texture,
    pub rects: Option<Vec<PixelRegion>>,
}

impl DirtyTileState {
    pub fn new() -> Self {
        Self {
            previous: Vec::new(),
            retained: None,
            high_change_streak: 0,
            tracking_cooldown: 0,
        }
    }

    pub fn prepare(
        &mut self,
        descriptors: &Descriptors,
        commands: &CommandList,
        viewport: PixelRegion,
        format: wgpu::TextureFormat,
        sample_count: u32,
        clear: wgpu::Color,
    ) -> DirtyDecision {
        debug_assert!(sample_count > 1);
        let retained = self.retained.get_or_insert_with(|| {
            let size = wgpu::Extent3d {
                width: viewport.width(),
                height: viewport.height(),
                depth_or_array_layers: 1,
            };
            let texture = |label, sample_count, usage| {
                descriptors.device.create_texture(&wgpu::TextureDescriptor {
                    label: create_debug_label!("{label}").as_deref(),
                    size,
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[format],
                })
            };
            RetainedFrame {
                multisampled: texture(
                    "Dirty-tile retained MSAA frame",
                    sample_count,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                ),
                resolved: texture(
                    "Dirty-tile retained resolve frame",
                    1,
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                ),
            }
        });

        if self.tracking_cooldown > 0 {
            self.tracking_cooldown -= 1;
            self.previous.clear();
            return DirtyDecision {
                multisampled: retained.multisampled.clone(),
                resolved: retained.resolved.clone(),
                rects: None,
            };
        }

        let fingerprint = Fingerprint::new(viewport, clear).build(commands);

        let can_reuse = !fingerprint.force_full && self.previous.len() == fingerprint.hashes.len();
        let dirty = if can_reuse {
            fingerprint
                .hashes
                .iter()
                .zip(&self.previous)
                .map(|(current, previous)| current != previous)
                .collect::<Vec<_>>()
        } else {
            vec![true; fingerprint.hashes.len()]
        };
        self.previous = if fingerprint.force_full {
            Vec::new()
        } else {
            fingerprint.hashes
        };

        let dirty_count = dirty.iter().filter(|dirty| **dirty).count();
        let use_partial =
            can_reuse && dirty_count * MAX_DIRTY_TILE_FRACTION_DENOMINATOR <= dirty.len();
        let rects = use_partial
            .then(|| coalesce_tiles(&dirty, fingerprint.tiles_x, fingerprint.tiles_y, viewport));
        if use_partial {
            self.high_change_streak = 0;
        } else if fingerprint.force_full {
            self.high_change_streak = 0;
            self.tracking_cooldown = TRACKING_COOLDOWN_FRAMES;
        } else if can_reuse {
            self.high_change_streak += 1;
            if self.high_change_streak >= HIGH_CHANGE_STREAK_LIMIT {
                self.high_change_streak = 0;
                self.tracking_cooldown = TRACKING_COOLDOWN_FRAMES;
            }
        }
        DirtyDecision {
            multisampled: retained.multisampled.clone(),
            resolved: retained.resolved.clone(),
            rects,
        }
    }
}

struct Fingerprint {
    viewport: PixelRegion,
    tiles_x: u32,
    tiles_y: u32,
    hashes: Vec<u64>,
    range_xor: Vec<u64>,
    command_index: u64,
    force_full: bool,
    mask_depth: u32,
}

impl Fingerprint {
    fn new(viewport: PixelRegion, clear: wgpu::Color) -> Self {
        let tiles_x = viewport.width().div_ceil(TILE_SIZE);
        let tiles_y = viewport.height().div_ceil(TILE_SIZE);
        let mut clear_hash = 0xcbf29ce484222325;
        for component in [clear.r, clear.g, clear.b, clear.a] {
            mix(&mut clear_hash, component.to_bits());
        }
        Self {
            viewport,
            tiles_x,
            tiles_y,
            hashes: vec![clear_hash; (tiles_x * tiles_y) as usize],
            range_xor: vec![0; ((tiles_x + 1) * (tiles_y + 1)) as usize],
            command_index: 0,
            force_full: false,
            mask_depth: 0,
        }
    }

    fn build(mut self, commands: &CommandList) -> Self {
        self.commands(commands);
        self.materialize_ranges();
        self
    }

    fn commands(&mut self, commands: &CommandList) {
        for command in &commands.commands {
            match command {
                Command::RenderBitmap {
                    bitmap,
                    transform,
                    smoothing,
                    pixel_snapping,
                } => {
                    let texture = as_texture(bitmap);
                    let mut matrix = transform.matrix;
                    pixel_snapping.apply(&mut matrix);
                    matrix *= Matrix::scale(
                        texture.texture.width() as f32,
                        texture.texture.height() as f32,
                    );
                    let mut signature = command_signature(1, matrix, transform.color_transform);
                    mix(&mut signature, arc_id(&bitmap.0) as u64);
                    mix(&mut signature, texture.generation.get());
                    mix(&mut signature, u64::from(*smoothing));
                    mix(&mut signature, snapping_id(*pixel_snapping));
                    self.add_transformed(signature, matrix, [0.0, 0.0, 1.0, 1.0]);
                }
                Command::RenderStage3D { .. } => self.force_full = true,
                Command::RenderShape { shape, transform } => {
                    let mesh = as_mesh(shape);
                    if let Some(bounds) = mesh.bounds {
                        let mut signature =
                            command_signature(2, transform.matrix, transform.color_transform);
                        mix(&mut signature, arc_id(&shape.0) as u64);
                        for draw in &mesh.draws {
                            if let DrawType::Bitmap { bitmap, .. } = &draw.draw_type {
                                let texture = as_texture(bitmap);
                                mix(&mut signature, arc_id(&bitmap.0) as u64);
                                mix(&mut signature, texture.generation.get());
                            }
                        }
                        self.add_transformed(signature, transform.matrix, bounds);
                    }
                }
                Command::DrawRect { color, matrix } => self.add_transformed(
                    command_signature(3, *matrix, ColorTransform::multiply_from(*color)),
                    *matrix,
                    [0.0, 0.0, 1.0, 1.0],
                ),
                Command::DrawLine { color, matrix } => self.add_transformed(
                    command_signature(4, *matrix, ColorTransform::multiply_from(*color)),
                    *matrix,
                    [0.0, 0.0, 1.0, 1.0],
                ),
                Command::DrawLineRect { color, matrix } => self.add_transformed(
                    command_signature(5, *matrix, ColorTransform::multiply_from(*color)),
                    *matrix,
                    [0.0, 0.0, 1.0, 1.0],
                ),
                Command::PushMask => {
                    self.mask_depth += 1;
                    self.force_full |= self.mask_depth >= 0x80;
                    self.add_region(6, viewport_bounds(self.viewport));
                }
                Command::ActivateMask => self.add_region(7, viewport_bounds(self.viewport)),
                Command::DeactivateMask => self.add_region(8, viewport_bounds(self.viewport)),
                Command::PopMask => {
                    self.add_region(9, viewport_bounds(self.viewport));
                    self.mask_depth = self.mask_depth.saturating_sub(1);
                }
                Command::Blend {
                    commands,
                    blend_mode,
                    bounds,
                } => {
                    let tag = match BlendType::from(blend_mode.clone()) {
                        BlendType::Trivial(TrivialBlend::Normal) => 10,
                        BlendType::Trivial(TrivialBlend::Add) => 11,
                        BlendType::Trivial(TrivialBlend::Subtract) => 12,
                        BlendType::Trivial(TrivialBlend::Screen) => 13,
                        BlendType::Complex(ComplexBlend::Multiply) => 14,
                        BlendType::Complex(ComplexBlend::Lighten) => 15,
                        BlendType::Complex(ComplexBlend::Darken) => 16,
                        BlendType::Complex(ComplexBlend::Difference) => 17,
                        BlendType::Complex(ComplexBlend::Invert) => 18,
                        BlendType::Complex(ComplexBlend::Alpha) => 19,
                        BlendType::Complex(ComplexBlend::Erase) => 20,
                        BlendType::Complex(ComplexBlend::Overlay) => 21,
                        BlendType::Complex(ComplexBlend::HardLight) => 22,
                        BlendType::Shader(_) => {
                            self.force_full = true;
                            23
                        }
                    };
                    let signature = self.nested_signature(commands, tag);
                    self.add_region(signature, pixel_bounds(*bounds));
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                    bounds,
                } => {
                    let mut signature = self.nested_signature(maskee_commands, 24);
                    let mask_signature = self.nested_signature(mask_commands, 25);
                    mix(&mut signature, mask_signature);
                    self.add_region(signature, pixel_bounds(*bounds));
                }
            }
        }
    }

    fn nested_signature(&mut self, commands: &CommandList, seed: u64) -> u64 {
        let mut nested = Fingerprint {
            viewport: self.viewport,
            tiles_x: 1,
            tiles_y: 1,
            hashes: vec![seed],
            range_xor: vec![0; 4],
            command_index: 0,
            force_full: false,
            mask_depth: 0,
        };
        nested.commands(commands);
        nested.materialize_ranges();
        self.force_full |= nested.force_full;
        nested.hashes[0]
    }

    fn add_transformed(&mut self, signature: u64, matrix: Matrix, bounds: [f32; 4]) {
        let points = [
            transform_point(matrix, bounds[0], bounds[1]),
            transform_point(matrix, bounds[2], bounds[1]),
            transform_point(matrix, bounds[0], bounds[3]),
            transform_point(matrix, bounds[2], bounds[3]),
        ];
        let mut output = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for [x, y] in points {
            output[0] = output[0].min(x);
            output[1] = output[1].min(y);
            output[2] = output[2].max(x);
            output[3] = output[3].max(y);
        }
        self.add_region(
            signature,
            [
                output[0] - 2.0,
                output[1] - 2.0,
                output[2] + 2.0,
                output[3] + 2.0,
            ],
        );
    }

    fn add_region(&mut self, mut signature: u64, bounds: [f32; 4]) {
        let left = bounds[0].floor().max(self.viewport.x_min as f32);
        let top = bounds[1].floor().max(self.viewport.y_min as f32);
        let right = bounds[2].ceil().min(self.viewport.x_max as f32);
        let bottom = bounds[3].ceil().min(self.viewport.y_max as f32);
        if !(left < right && top < bottom) {
            return;
        }
        let x0 = ((left as u32 - self.viewport.x_min) / TILE_SIZE).min(self.tiles_x - 1);
        let y0 = ((top as u32 - self.viewport.y_min) / TILE_SIZE).min(self.tiles_y - 1);
        let x1 = (((right as u32).saturating_sub(1) - self.viewport.x_min) / TILE_SIZE)
            .min(self.tiles_x - 1);
        let y1 = (((bottom as u32).saturating_sub(1) - self.viewport.y_min) / TILE_SIZE)
            .min(self.tiles_y - 1);
        mix(&mut signature, self.command_index);
        self.command_index += 1;
        let stride = self.tiles_x + 1;
        for (x, y) in [(x0, y0), (x1 + 1, y0), (x0, y1 + 1), (x1 + 1, y1 + 1)] {
            self.range_xor[(y * stride + x) as usize] ^= signature;
        }
    }

    fn materialize_ranges(&mut self) {
        let stride = self.tiles_x + 1;
        for y in 0..self.tiles_y {
            for x in 0..self.tiles_x {
                let index = (y * stride + x) as usize;
                let mut value = self.range_xor[index];
                if x > 0 {
                    value ^= self.range_xor[index - 1];
                }
                if y > 0 {
                    value ^= self.range_xor[index - stride as usize];
                }
                if x > 0 && y > 0 {
                    value ^= self.range_xor[index - stride as usize - 1];
                }
                self.range_xor[index] = value;
                mix(&mut self.hashes[(y * self.tiles_x + x) as usize], value);
            }
        }
    }
}

fn coalesce_tiles(
    dirty: &[bool],
    tiles_x: u32,
    tiles_y: u32,
    viewport: PixelRegion,
) -> Vec<PixelRegion> {
    let mut used = vec![false; dirty.len()];
    let mut rects = Vec::new();
    for y in 0..tiles_y {
        for x in 0..tiles_x {
            let index = (y * tiles_x + x) as usize;
            if !dirty[index] || used[index] {
                continue;
            }
            let mut width = 1;
            while x + width < tiles_x {
                let next = (y * tiles_x + x + width) as usize;
                if !dirty[next] || used[next] {
                    break;
                }
                width += 1;
            }
            let mut height = 1;
            'rows: while y + height < tiles_y {
                for offset in 0..width {
                    let next = ((y + height) * tiles_x + x + offset) as usize;
                    if !dirty[next] || used[next] {
                        break 'rows;
                    }
                }
                height += 1;
            }
            for row in y..y + height {
                for column in x..x + width {
                    used[(row * tiles_x + column) as usize] = true;
                }
            }
            let px = x * TILE_SIZE;
            let py = y * TILE_SIZE;
            rects.push(PixelRegion::for_region(
                px,
                py,
                (width * TILE_SIZE).min(viewport.width() - px),
                (height * TILE_SIZE).min(viewport.height() - py),
            ));
        }
    }
    rects
}

fn command_signature(tag: u64, matrix: Matrix, color: ColorTransform) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    mix(&mut hash, tag);
    for value in [matrix.a, matrix.b, matrix.c, matrix.d] {
        mix(&mut hash, value.to_bits() as u64);
    }
    mix(&mut hash, (matrix.tx.to_pixels() as f32).to_bits() as u64);
    mix(&mut hash, (matrix.ty.to_pixels() as f32).to_bits() as u64);
    for value in color
        .mult_rgba_normalized()
        .into_iter()
        .chain(color.add_rgba_normalized())
    {
        mix(&mut hash, value.to_bits() as u64);
    }
    hash
}

fn transform_point(matrix: Matrix, x: f32, y: f32) -> [f32; 2] {
    [
        matrix.a * x + matrix.c * y + matrix.tx.to_pixels() as f32,
        matrix.b * x + matrix.d * y + matrix.ty.to_pixels() as f32,
    ]
}

fn viewport_bounds(viewport: PixelRegion) -> [f32; 4] {
    [
        viewport.x_min as f32,
        viewport.y_min as f32,
        viewport.x_max as f32,
        viewport.y_max as f32,
    ]
}

fn pixel_bounds(region: PixelRegion) -> [f32; 4] {
    [
        region.x_min as f32,
        region.y_min as f32,
        region.x_max as f32,
        region.y_max as f32,
    ]
}

fn snapping_id(snapping: PixelSnapping) -> u64 {
    match snapping {
        PixelSnapping::Always => 1,
        PixelSnapping::Auto => 2,
        PixelSnapping::Never => 3,
    }
}

fn arc_id<T: ?Sized>(arc: &Arc<T>) -> usize {
    Arc::as_ptr(arc) as *const () as usize
}

fn mix(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x100000001b3);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruffle_render::commands::{CommandHandler, RenderBlendMode};
    use swf::{BlendMode, Color, Twips};

    #[test]
    fn coalesces_rectangles_without_covering_clean_tiles() {
        let dirty = [true, true, false, true, true, true, false, true];
        let rects = coalesce_tiles(&dirty, 4, 2, PixelRegion::for_whole_size(128, 64));
        assert_eq!(
            rects,
            vec![
                PixelRegion::for_region(0, 0, 64, 64),
                PixelRegion::for_region(96, 0, 32, 64),
            ]
        );
    }

    #[test]
    fn clips_edge_tiles() {
        let rects = coalesce_tiles(
            &[false, true, false, true],
            2,
            2,
            PixelRegion::for_whole_size(50, 45),
        );
        assert_eq!(rects, vec![PixelRegion::for_region(32, 0, 18, 45)]);
    }

    #[test]
    fn nested_changes_invalidate_the_entire_effect_bounds() {
        let make = |x| {
            let mut child = CommandList::new();
            child.draw_rect(
                Color::WHITE,
                Matrix {
                    tx: Twips::from_pixels(x),
                    a: 8.0,
                    d: 8.0,
                    ..Default::default()
                },
            );
            let mut commands = CommandList::new();
            commands.blend(
                child,
                RenderBlendMode::Builtin(BlendMode::Normal),
                PixelRegion::for_whole_size(64, 32),
            );
            Fingerprint::new(PixelRegion::for_whole_size(64, 32), wgpu::Color::BLACK)
                .build(&commands)
                .hashes
        };

        let first = make(0.0);
        let moved = make(4.0);
        assert_eq!(first[0], first[1]);
        assert_ne!(first[0], moved[0]);
        assert_ne!(first[1], moved[1]);
    }
}
