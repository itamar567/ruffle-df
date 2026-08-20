use crate::tag_utils::{SwfMovie, SwfSlice};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;
use swf::avm1::read::Reader;
use swf::avm1::types::Action;

const MAX_BLOCKS: usize = 128;
const MAX_ACTIONS: usize = 32_768;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    movie: *const SwfMovie,
    start: usize,
    len: usize,
    version: u8,
}

pub(crate) struct CachedAction {
    pub action: Action<'static>,
    pub next: usize,
    pub jump_target: Option<usize>,
}

pub(crate) struct CachedBlock {
    pub movie: Arc<SwfMovie>,
    pub actions: Vec<CachedAction>,
}

pub(crate) struct ActionCache {
    entries: HashMap<CacheKey, Rc<CachedBlock>>,
    lru: VecDeque<CacheKey>,
    action_count: usize,
}

impl ActionCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            action_count: 0,
        }
    }

    pub(crate) fn get_or_decode(
        &mut self,
        code: &SwfSlice,
        version: u8,
    ) -> Option<Rc<CachedBlock>> {
        if code.movie.version() != version
            || !code.movie.is_data_complete()
            || code.end > code.movie.data().len()
        {
            return None;
        }

        let key = CacheKey {
            movie: Arc::as_ptr(&code.movie),
            start: code.start,
            len: code.len(),
            version,
        };
        if let Some(block) = self.entries.get(&key).cloned() {
            self.touch(key);
            return Some(block);
        }

        let block = decode(code, version)?;
        if block.actions.is_empty() || block.actions.len() > MAX_ACTIONS {
            return None;
        }

        let block = Rc::new(block);
        self.action_count += block.actions.len();
        self.entries.insert(key, block.clone());
        self.touch(key);
        self.evict();
        Some(block)
    }

    fn touch(&mut self, key: CacheKey) {
        if let Some(index) = self.lru.iter().position(|candidate| *candidate == key) {
            self.lru.remove(index);
        }
        self.lru.push_back(key);
    }

    fn evict(&mut self) {
        while self.entries.len() > MAX_BLOCKS || self.action_count > MAX_ACTIONS {
            let Some(key) = self.lru.pop_front() else {
                break;
            };
            if let Some(block) = self.entries.remove(&key) {
                self.action_count -= block.actions.len();
            }
        }
    }
}

fn decode(code: &SwfSlice, version: u8) -> Option<CachedBlock> {
    let movie_data = code.movie.data();
    let mut reader = Reader::new(&movie_data[code.start..code.end], version);
    let mut actions = Vec::new();
    let mut offsets = HashMap::new();

    while !reader.get_ref().is_empty() {
        let action_start = movie_data.len() - reader.get_ref().len();
        let action_start = action_start.checked_sub(code.start)?;
        let action = reader.read_action().ok()?;
        let next = movie_data.len() - reader.get_ref().len() - code.start;
        let unsupported = matches!(
            action,
            Action::Try(_) | Action::WaitForFrame(_) | Action::WaitForFrame2(_) | Action::With(_)
        );
        if unsupported {
            return None;
        }

        offsets.insert(action_start, actions.len());
        let jump_offset = match &action {
            Action::If(action) => Some(action.offset),
            Action::Jump(action) => Some(action.offset),
            _ => None,
        };
        let jump_target = if let Some(jump_offset) = jump_offset {
            let target = (code.start + next).checked_add_signed(jump_offset as isize)?;
            if target < code.start || target > code.end {
                return None;
            }
            Some(target - code.start)
        } else {
            None
        };
        actions.push(CachedAction {
            // The movie is held by CachedBlock, so these references remain valid for the
            // lifetime of every cached instruction. They are never exposed outside the cache.
            action: unsafe { std::mem::transmute(action) },
            next,
            jump_target,
        });

        if matches!(actions.last().map(|a| &a.action), Some(Action::End)) {
            break;
        }
    }

    let implicit_return = actions.len();
    for action in &mut actions {
        if let Some(target) = action.jump_target {
            action.jump_target = Some(*offsets.get(&target).unwrap_or(&implicit_return));
        }
    }

    Some(CachedBlock {
        movie: code.movie.clone(),
        actions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tag_utils::SwfMovie;

    fn code(bytes: &[u8]) -> SwfSlice {
        let movie = Arc::new(SwfMovie::fake_with_compressed_data(8, None, bytes.to_vec()));
        SwfSlice::from(movie)
    }

    #[test]
    fn decodes_push_and_end_once() {
        let mut cache = ActionCache::new();
        let block = cache.get_or_decode(&code(&[0x96, 2, 0, 0, 0, 0]), 8);
        assert!(block.is_some());
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn jump_target_is_precomputed() {
        let mut cache = ActionCache::new();
        let block = cache
            .get_or_decode(&code(&[0x96, 2, 0, 5, 1, 0x9D, 2, 0, 1, 0, 0x17, 0]), 8)
            .unwrap();
        assert_eq!(block.actions[1].jump_target, Some(3));
    }

    #[test]
    fn malformed_and_nested_control_flow_fall_back() {
        let mut cache = ActionCache::new();
        assert!(cache.get_or_decode(&code(&[0x96, 1, 0]), 8).is_none());
        assert!(cache.get_or_decode(&code(&[0x94, 2, 0, 0, 0]), 8).is_none());
    }

    #[test]
    fn cache_is_bounded() {
        let mut cache = ActionCache::new();
        for _ in 0..(MAX_BLOCKS + 1) {
            assert!(cache.get_or_decode(&code(&[0]), 8).is_some());
        }
        assert_eq!(cache.entries.len(), MAX_BLOCKS);
    }
}
