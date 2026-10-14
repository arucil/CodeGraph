// Copyright 2025-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Workspace load slots shared by connections to the socket engine.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

pub(super) struct WorkspaceRegistry<T> {
    slots: Mutex<HashMap<PathBuf, Weak<T>>>,
}

impl<T> Default for WorkspaceRegistry<T> {
    fn default() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }
}

impl<T: Default> WorkspaceRegistry<T> {
    pub(super) fn get_or_create(&self, workspace: PathBuf) -> Arc<T> {
        let mut slots = self.slots.lock().expect("workspace registry poisoned");
        // Connections, not this cache, own the graph and file watcher. A weak
        // slot also keeps simultaneous cold attaches on the same OnceCell.
        slots.retain(|_, slot| slot.strong_count() != 0);
        if let Some(slot) = slots.get(&workspace).and_then(Weak::upgrade) {
            return slot;
        }
        let slot = Arc::new(T::default());
        slots.insert(workspace, Arc::downgrade(&slot));
        slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn last_client_releases_workspace() {
        let registry = WorkspaceRegistry::<Vec<u8>>::default();
        let slot = registry.get_or_create(PathBuf::from("project"));
        let weak = Arc::downgrade(&slot);

        drop(slot);

        assert!(
            weak.upgrade().is_none(),
            "closed workspaces must not remain resident"
        );
    }

    #[test]
    fn another_client_keeps_the_same_workspace_alive() {
        let registry = WorkspaceRegistry::<Vec<u8>>::default();
        let workspace = PathBuf::from("project");
        let first = registry.get_or_create(workspace.clone());
        let second = registry.get_or_create(workspace.clone());
        assert!(Arc::ptr_eq(&first, &second));
        let weak = Arc::downgrade(&first);
        drop(first);

        let third = registry.get_or_create(workspace);
        assert!(Arc::ptr_eq(&second, &third));
        drop(second);
        assert!(weak.upgrade().is_some());
        drop(third);
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn last_client_releases_initialized_backend() {
        let registry = WorkspaceRegistry::<tokio::sync::OnceCell<Arc<Vec<u8>>>>::default();
        let slot = registry.get_or_create(PathBuf::from("project"));
        let backend = Arc::clone(slot.get_or_init(|| async { Arc::new(vec![1]) }).await);
        let weak = Arc::downgrade(&backend);
        drop(backend);
        assert!(weak.upgrade().is_some());

        drop(slot);

        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn simultaneous_cold_clients_initialize_only_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let registry =
            Arc::new(WorkspaceRegistry::<tokio::sync::OnceCell<Arc<Vec<u8>>>>::default());
        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let loads = Arc::new(AtomicUsize::new(0));
        let mut clients = Vec::new();
        for _ in 0..8 {
            let registry = Arc::clone(&registry);
            let barrier = Arc::clone(&barrier);
            let loads = Arc::clone(&loads);
            clients.push(tokio::spawn(async move {
                let slot = registry.get_or_create(PathBuf::from("project"));
                barrier.wait().await;
                let backend = Arc::clone(
                    slot.get_or_init(|| async {
                        loads.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        Arc::new(vec![1])
                    })
                    .await,
                );
                (slot, backend)
            }));
        }
        let mut connections = Vec::new();
        for client in clients {
            connections.push(client.await.unwrap());
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(connections
            .iter()
            .all(|(_, backend)| Arc::ptr_eq(backend, &connections[0].1)));
        let weak = Arc::downgrade(&connections[0].1);

        drop(connections);

        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn releasing_one_workspace_does_not_release_another() {
        let registry = WorkspaceRegistry::<Vec<u8>>::default();
        let first = registry.get_or_create(PathBuf::from("first"));
        let second = registry.get_or_create(PathBuf::from("second"));
        let first_weak = Arc::downgrade(&first);
        drop(first);

        assert!(first_weak.upgrade().is_none());
        assert!(Arc::ptr_eq(
            &second,
            &registry.get_or_create(PathBuf::from("second"))
        ));
    }

    #[test]
    fn reconnect_after_release_gets_a_fresh_slot() {
        let registry = WorkspaceRegistry::<Vec<u8>>::default();
        let workspace = PathBuf::from("project");
        let first = registry.get_or_create(workspace.clone());
        let old = Arc::downgrade(&first);
        drop(first);

        let reopened = registry.get_or_create(workspace);
        assert!(old.upgrade().is_none());
        assert_eq!(Arc::strong_count(&reopened), 1);
    }

    #[test]
    fn simultaneous_attaches_share_one_load_slot() {
        let registry = Arc::new(WorkspaceRegistry::<Vec<u8>>::default());
        let barrier = Arc::new(Barrier::new(8));
        let clients: Vec<_> = (0..8)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    registry.get_or_create(PathBuf::from("project"))
                })
            })
            .collect();
        let slots: Vec<_> = clients.into_iter().map(|c| c.join().unwrap()).collect();

        assert!(slots.iter().all(|slot| Arc::ptr_eq(slot, &slots[0])));
        let weak = Arc::downgrade(&slots[0]);
        drop(slots);
        assert!(weak.upgrade().is_none());
    }
}
