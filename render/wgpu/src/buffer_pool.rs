use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
use ruffle_render::bitmap::PixelRegion;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};

const MAX_TEXTURE_POOL_KEYS: usize = 256;
const MAX_TEXTURES_PER_KEY: usize = 2;
const MAX_TEXTURE_POOL_RETAINED_BYTES: usize = 256 * 1024 * 1024;
const MAX_GLOBALS: usize = 256;

struct PoolInner<T> {
    available: Vec<T>,
    retained_capacity: usize,
}

type SharedPool<T> = Mutex<PoolInner<T>>;
type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

#[derive(Debug)]
struct CacheEntry<Value> {
    value: Value,
    last_access: u64,
    weight: usize,
}

#[derive(Debug)]
struct BoundedCache<Key, Value> {
    entries: FnvHashMap<Key, CacheEntry<Value>>,
    capacity: usize,
    max_weight: usize,
    total_weight: usize,
    access: u64,
}

impl<Key: Clone + Eq + Hash, Value> BoundedCache<Key, Value> {
    fn new(capacity: usize) -> Self {
        Self::with_max_weight(capacity, usize::MAX)
    }

    fn with_max_weight(capacity: usize, max_weight: usize) -> Self {
        assert!(capacity > 0);
        Self {
            entries: FnvHashMap::default(),
            capacity,
            max_weight,
            total_weight: 0,
            access: 0,
        }
    }

    fn get_or_insert_with(&mut self, key: Key, create: impl FnOnce() -> Value) -> &mut Value {
        self.get_or_insert_with_weight(key, 0, create)
    }

    fn get_or_insert_with_weight(
        &mut self,
        key: Key,
        weight: usize,
        create: impl FnOnce() -> Value,
    ) -> &mut Value {
        assert!(weight <= self.max_weight);
        self.access = self.access.wrapping_add(1);
        if self.entries.contains_key(&key) {
            let entry = self.entries.get_mut(&key).unwrap();
            assert_eq!(entry.weight, weight);
            entry.last_access = self.access;
            return &mut entry.value;
        }

        while self.entries.len() >= self.capacity
            || self.total_weight > self.max_weight - weight
        {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(key, _)| key.clone())
                .unwrap();
            let removed = self.entries.remove(&oldest).unwrap();
            self.total_weight -= removed.weight;
        }

        self.total_weight += weight;
        &mut self
            .entries
            .entry(key)
            .or_insert_with(|| CacheEntry {
                value: create(),
                last_access: self.access,
                weight,
            })
            .value
    }
}

#[derive(Debug)]
pub struct TexturePool {
    pools:
        BoundedCache<TextureKey, BufferPool<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>>,
    globals_cache: BoundedCache<PixelRegion, Arc<Globals>>,
}

impl Default for TexturePool {
    fn default() -> Self {
        Self::new()
    }
}

impl TexturePool {
    pub fn new() -> Self {
        Self {
            pools: BoundedCache::with_max_weight(
                MAX_TEXTURE_POOL_KEYS,
                MAX_TEXTURE_POOL_RETAINED_BYTES,
            ),
            globals_cache: BoundedCache::new(MAX_GLOBALS),
        }
    }

    pub fn get_texture(
        &mut self,
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible> {
        let key = TextureKey {
            size,
            usage,
            format,
            sample_count,
        };
        let Some((retained_capacity, retained_bytes)) = key.retained_capacity_and_bytes() else {
            return Self::create_texture_pool(key, 0).take(descriptors, AlwaysCompatible);
        };
        let pool = self
            .pools
            .get_or_insert_with_weight(key, retained_bytes, || {
                Self::create_texture_pool(key, retained_capacity)
            });
        pool.take(descriptors, AlwaysCompatible)
    }

    fn create_texture_pool(
        key: TextureKey,
        retained_capacity: usize,
    ) -> BufferPool<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible> {
        let label = if cfg!(feature = "render_debug_labels") {
            use std::sync::atomic::{AtomicU32, Ordering};
            static ID_COUNT: AtomicU32 = AtomicU32::new(0);
            let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
            create_debug_label!("Pooled texture {}", id)
        } else {
            None
        };
        BufferPool::with_retained_capacity(
            Box::new(move |descriptors, _description| {
                let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
                    label: label.as_deref(),
                    size: key.size,
                    mip_level_count: 1,
                    sample_count: key.sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format: key.format,
                    view_formats: &[key.format],
                    usage: key.usage,
                });
                let view = texture.create_view(&Default::default());
                (texture, view)
            }),
            retained_capacity,
        )
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport: PixelRegion,
    ) -> Arc<Globals> {
        self.globals_cache
            .get_or_insert_with(viewport, || {
                Arc::new(Globals::new(
                    &descriptors.device,
                    &descriptors.bind_layouts.globals,
                    viewport,
                ))
            })
            .clone()
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct TextureKey {
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    sample_count: u32,
}

impl TextureKey {
    fn estimated_bytes(self) -> usize {
        let (block_width, block_height) = self.format.block_dimensions();
        let block_size = self
            .format
            .block_copy_size(None)
            .map(u64::from)
            .unwrap_or(u64::MAX);
        let width_blocks = u64::from(self.size.width.div_ceil(block_width));
        let height_blocks = u64::from(self.size.height.div_ceil(block_height));

        width_blocks
            .saturating_mul(height_blocks)
            .saturating_mul(u64::from(self.size.depth_or_array_layers))
            .saturating_mul(u64::from(self.sample_count))
            .saturating_mul(block_size)
            .min(usize::MAX as u64) as usize
    }

    fn retained_capacity_and_bytes(self) -> Option<(usize, usize)> {
        let texture_bytes = self.estimated_bytes();
        if texture_bytes == 0 {
            return None;
        }

        let capacity =
            (MAX_TEXTURE_POOL_RETAINED_BYTES / texture_bytes).min(MAX_TEXTURES_PER_KEY);
        (capacity > 0).then_some((capacity, texture_bytes.saturating_mul(capacity)))
    }
}

pub trait BufferDescription: Clone + Debug {
    type Cost: Ord;

    /// If the potential buffer represented by this description (`self`)
    /// fits another existing buffer and its description (`other`),
    /// return the cost to use that buffer instead of making a new one.
    ///
    /// Cost is an arbitrary unit, but lower is better.
    /// None means that the other buffer cannot be used in place of this one.
    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost>;
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
        Some(())
    }
}

pub struct BufferPool<Type, Description: BufferDescription> {
    available: Arc<SharedPool<(Type, Description)>>,
    constructor: Constructor<Type, Description>,
}

impl<Type, Description: BufferDescription> Debug for BufferPool<Type, Description> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish()
    }
}

impl<Type, Description: BufferDescription> BufferPool<Type, Description> {
    pub fn new(constructor: Constructor<Type, Description>) -> Self {
        Self::with_retained_capacity(constructor, usize::MAX)
    }

    pub fn with_retained_capacity(
        constructor: Constructor<Type, Description>,
        retained_capacity: usize,
    ) -> Self {
        Self {
            available: Arc::new(Mutex::new(PoolInner {
                available: Vec::new(),
                retained_capacity,
            })),
            constructor,
        }
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self
            .available
            .lock()
            .expect("Should not be able to lock recursively");
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.available.len() {
            if let Some(cost) = description.cost_to_use(&guard.available[i].1) {
                if let Some(best) = &mut best {
                    if best.0 > cost {
                        *best = (cost, i);
                    }
                } else if best.is_none() {
                    best = Some((cost, i));
                }
            }
        }

        let (item, used_description) = if let Some((_, best)) = best {
            guard.available.swap_remove(best)
        } else {
            let item = (self.constructor)(descriptors, &description);
            (item, description)
        };
        PoolEntry {
            item: Some(item),
            description: used_description,
            pool: Arc::downgrade(&self.available),
        }
    }
}

pub struct PoolEntry<Type, Description: BufferDescription> {
    item: Option<Type>,
    description: Description,
    pool: Weak<SharedPool<(Type, Description)>>,
}

impl<Type, Description: BufferDescription> Debug for PoolEntry<Type, Description>
where
    Type: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PoolEntry").field(&self.item).finish()
    }
}

impl<Type, Description: BufferDescription> Drop for PoolEntry<Type, Description> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take()
            && let Some(pool) = self.pool.upgrade()
        {
            let mut pool = pool.lock().expect("Should not be able to lock recursively");
            if pool.available.len() < pool.retained_capacity {
                pool.available.push((item, self.description.clone()));
            }
        }
    }
}

impl<Type, Description: BufferDescription> Deref for PoolEntry<Type, Description> {
    type Target = Type;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Item should exist until dropped")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_cache_rejects_zero_capacity() {
        assert!(std::panic::catch_unwind(|| BoundedCache::<u8, u8>::new(0)).is_err());
    }

    #[test]
    fn bounded_cache_evicts_least_recently_used_entry() {
        let mut cache = BoundedCache::new(2);
        cache.get_or_insert_with(1, || "one");
        cache.get_or_insert_with(2, || "two");
        cache.get_or_insert_with(3, || "three");

        assert!(!cache.entries.contains_key(&1));
        assert!(cache.entries.contains_key(&2));
        assert!(cache.entries.contains_key(&3));
    }

    #[test]
    fn bounded_cache_refreshes_existing_entry() {
        let mut cache = BoundedCache::new(2);
        cache.get_or_insert_with(1, || "one");
        cache.get_or_insert_with(2, || "two");
        cache.get_or_insert_with(1, || unreachable!());
        cache.get_or_insert_with(3, || "three");

        assert!(cache.entries.contains_key(&1));
        assert!(!cache.entries.contains_key(&2));
        assert!(cache.entries.contains_key(&3));
    }

    #[test]
    fn weighted_cache_evicts_oldest_entries_to_fit_budget() {
        let mut cache = BoundedCache::with_max_weight(3, 10);
        cache.get_or_insert_with_weight(1, 4, || "one");
        cache.get_or_insert_with_weight(2, 4, || "two");
        cache.get_or_insert_with_weight(1, 4, || unreachable!());
        cache.get_or_insert_with_weight(3, 4, || "three");

        assert!(cache.entries.contains_key(&1));
        assert!(!cache.entries.contains_key(&2));
        assert!(cache.entries.contains_key(&3));
        assert_eq!(cache.total_weight, 8);
    }

    #[test]
    fn texture_bytes_account_for_blocks_layers_and_samples() {
        let key = |width, height, layers, format, sample_count| TextureKey {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            sample_count,
        };

        assert_eq!(
            key(10, 20, 2, wgpu::TextureFormat::Rgba8Unorm, 4).estimated_bytes(),
            6_400
        );
        assert_eq!(
            key(5, 5, 1, wgpu::TextureFormat::Bc1RgbaUnorm, 1).estimated_bytes(),
            32
        );
    }

    #[test]
    fn texture_retention_reserves_worst_case_bytes_per_key() {
        let key = |width, height| TextureKey {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            sample_count: 1,
        };

        assert_eq!(
            key(4_096, 4_096).retained_capacity_and_bytes(),
            Some((2, 128 * 1024 * 1024))
        );
        assert_eq!(
            key(8_192, 8_192).retained_capacity_and_bytes(),
            Some((1, 256 * 1024 * 1024))
        );
        assert_eq!(key(16_384, 16_384).retained_capacity_and_bytes(), None);
    }

    #[test]
    fn pool_entry_limits_returned_items() {
        let pool = Arc::new(Mutex::new(PoolInner {
            available: Vec::new(),
            retained_capacity: 2,
        }));
        for item in 0..3 {
            drop(PoolEntry {
                item: Some(item),
                description: AlwaysCompatible,
                pool: Arc::downgrade(&pool),
            });
        }

        assert_eq!(pool.lock().unwrap().available.len(), 2);
    }

    #[test]
    fn pool_entry_drops_item_after_pool_is_evicted() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct DropFlag(Rc<Cell<bool>>);

        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let dropped = Rc::new(Cell::new(false));
        let pool = Arc::new(Mutex::new(PoolInner {
            available: Vec::new(),
            retained_capacity: 1,
        }));
        let entry = PoolEntry {
            item: Some(DropFlag(Rc::clone(&dropped))),
            description: AlwaysCompatible,
            pool: Arc::downgrade(&pool),
        };
        drop(pool);
        drop(entry);

        assert!(dropped.get());
    }
}
