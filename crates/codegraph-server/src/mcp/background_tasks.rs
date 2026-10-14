// Copyright 2025-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Background work that must not outlive its workspace backend.

use std::future::Future;
use std::sync::Mutex;
use tokio::task::JoinSet;

#[derive(Default)]
pub(super) struct BackgroundTasks {
    tasks: Mutex<JoinSet<()>>,
}

impl BackgroundTasks {
    pub(super) fn spawn(&self, future: impl Future<Output = ()> + Send + 'static) {
        let mut tasks = self
            .tasks
            .lock()
            .expect("workspace background tasks poisoned");
        while let Some(result) = tasks.try_join_next() {
            if let Err(error) = result {
                tracing::error!("Workspace background task failed: {error}");
            }
        }
        tasks.spawn(future);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::oneshot;

    struct DropNotice(Option<oneshot::Sender<()>>);

    impl Drop for DropNotice {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    #[tokio::test]
    async fn dropping_workspace_cancels_work_and_releases_captured_resources() {
        let tasks = BackgroundTasks::default();
        let (started, ready) = oneshot::channel();
        let (dropped, released) = oneshot::channel();
        tasks.spawn(async move {
            let _resource = DropNotice(Some(dropped));
            started.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        ready.await.unwrap();

        drop(tasks);

        tokio::time::timeout(std::time::Duration::from_secs(1), released)
            .await
            .expect("workspace work was not cancelled")
            .unwrap();
    }

    #[tokio::test]
    async fn live_backend_clone_keeps_its_work_running() {
        let tasks = Arc::new(BackgroundTasks::default());
        let remaining = Arc::clone(&tasks);
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let (started, ready) = oneshot::channel();
        let (dropped, released) = oneshot::channel();
        tasks.spawn(async move {
            let _resource = resource;
            let _notice = DropNotice(Some(dropped));
            started.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        ready.await.unwrap();

        drop(tasks);
        assert!(weak.upgrade().is_some());
        drop(remaining);

        tokio::time::timeout(std::time::Duration::from_secs(1), released)
            .await
            .expect("last backend clone did not release its work")
            .unwrap();
        assert!(weak.upgrade().is_none());
    }
}
