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
}

#[derive(Debug)]
struct BoundedCache<Key, Value> {
    entries: FnvHashMap<Key, CacheEntry<Value>>,
    capacity: usize,
    access: u64,
}

impl<Key: Clone + Eq + Hash, Value> BoundedCache<Key, Value> {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            entries: FnvHashMap::default(),
            capacity,
            access: 0,
        }
    }

    fn get_or_insert_with(&mut self, key: Key, create: impl FnOnce() -> Value) -> &mut Value {
        self.access = self.access.wrapping_add(1);
        if self.entries.contains_key(&key) {
            let entry = self.entries.get_mut(&key).unwrap();
            entry.last_access = self.access;
            return &mut entry.value;
        }

        if self.entries.len() >= self.capacity {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(key, _)| key.clone())
                .unwrap();
            self.entries.remove(&oldest);
        }

        &mut self
            .entries
            .entry(key)
            .or_insert_with(|| CacheEntry {
                value: create(),
                last_access: self.access,
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
            pools: BoundedCache::new(MAX_TEXTURE_POOL_KEYS),
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
        let pool = self.pools.get_or_insert_with(key, || {
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
                        size,
                        mip_level_count: 1,
                        sample_count,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        view_formats: &[format],
                        usage,
                    });
                    let view = texture.create_view(&Default::default());
                    (texture, view)
                }),
                MAX_TEXTURES_PER_KEY,
            )
        });
        pool.take(descriptors, AlwaysCompatible)
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
