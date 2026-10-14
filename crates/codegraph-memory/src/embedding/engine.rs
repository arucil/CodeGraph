// Copyright 2025-2026 Andrey Vasilevsky <anvanster@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Vector embedding engine
//!
//! High-level API for generating and caching embeddings.

use super::fastembed_embed::{CodeGraphEmbeddingModel, FastembedEmbedding};
use super::{Embedder, EmbeddingBackend};
use crate::error::Result;
use dashmap::DashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Vector embedding engine with caching
///
/// Wraps fastembed with a configurable model and DashMap cache for efficient repeated lookups.
pub struct VectorEngine {
    model: Arc<dyn Embedder>,
    cache: DashMap<String, Vec<f32>>,
    dimension: usize,
}

impl VectorEngine {
    /// Create VectorEngine with the default model (Jina Code V2)
    pub fn new(_extension_path: Option<&Path>) -> Result<Self> {
        Self::with_model(default_cache_dir(), CodeGraphEmbeddingModel::default())
    }

    /// Create VectorEngine with a specific embedding model
    pub fn with_model(cache_dir: PathBuf, model_type: CodeGraphEmbeddingModel) -> Result<Self> {
        let model = FastembedEmbedding::new(cache_dir, model_type)?;
        let dimension = model.dimension();

        log::info!(
            "VectorEngine ready ({}, {}d)",
            model_type.display_name(),
            dimension
        );

        Ok(Self {
            model: Arc::new(model),
            cache: DashMap::new(),
            dimension,
        })
    }

    /// Create a VectorEngine backed by a static (lookup-table) model loaded from
    /// a model2vec-format directory (`config.json` + `tokenizer.json` +
    /// `model.safetensors`). No ONNX — the fast indexing path.
    pub fn with_static_model(model_dir: &Path) -> Result<Self> {
        let model = super::static_embed::StaticEmbedding::from_pretrained(model_dir)?;
        let dimension = model.dimension();
        log::info!(
            "VectorEngine ready ({}, {}d, static)",
            model.model_name(),
            dimension
        );
        Ok(Self {
            model: Arc::new(model),
            cache: DashMap::new(),
            dimension,
        })
    }

    /// Build a VectorEngine from an `EmbeddingBackend` selection — the ONNX
    /// fastembed path or the static model2vec path.
    pub fn from_backend(cache_dir: PathBuf, backend: &EmbeddingBackend) -> Result<Self> {
        match backend {
            EmbeddingBackend::Fastembed(model) => Self::with_model(cache_dir, *model),
            EmbeddingBackend::Static(dir) => Self::with_static_model(dir),
        }
    }

    /// Reuse the loaded model, but give a workspace its own embedding cache.
    /// Sharing a whole VectorEngine would keep every disconnected workspace's
    /// input texts and vectors in the resident engine for its entire lifetime.
    pub fn with_shared_model(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            cache: DashMap::new(),
            dimension: self.dimension,
        }
    }

    /// Generate embedding with caching
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }

        // Generate and cache
        let embedding = self.model.embed(text)?;
        self.cache.insert(text.to_string(), embedding.clone());
        Ok(embedding)
    }

    /// Batch embed with caching
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Check cache for all texts
        let mut results: Vec<Option<Vec<f32>>> = texts
            .iter()
            .map(|text| self.cache.get(*text).map(|v| v.clone()))
            .collect();

        // Find uncached texts
        let uncached: Vec<(usize, &str)> = results
            .iter()
            .enumerate()
            .filter(|(_, cached)| cached.is_none())
            .map(|(i, _)| (i, texts[i]))
            .collect();

        if uncached.is_empty() {
            return Ok(results.into_iter().flatten().collect());
        }

        // Batch embed uncached texts
        let uncached_texts: Vec<&str> = uncached.iter().map(|(_, t)| *t).collect();
        let new_embeddings = self.model.embed_batch(&uncached_texts)?;

        // Update cache and results
        for ((idx, text), emb) in uncached.iter().zip(new_embeddings.into_iter()) {
            self.cache.insert(text.to_string(), emb.clone());
            results[*idx] = Some(emb);
        }

        Ok(results.into_iter().flatten().collect())
    }

    /// Cosine similarity between two embeddings
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get model display name (e.g. "Jina Code V2 (768d)")
    pub fn model_name(&self) -> &str {
        self.model.model_name()
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

/// Default cache directory for fastembed models
fn default_cache_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".codegraph")
        .join("fastembed_cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestEmbedder;

    impl Embedder for TestEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0])
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            texts.iter().map(|text| self.embed(text)).collect()
        }

        fn dimension(&self) -> usize {
            2
        }

        fn model_name(&self) -> &str {
            "test"
        }
    }

    fn shared_engine() -> VectorEngine {
        VectorEngine {
            model: Arc::new(TestEmbedder),
            cache: DashMap::new(),
            dimension: 2,
        }
    }

    #[test]
    fn workspace_cache_is_not_retained_by_shared_model() {
        let shared = shared_engine();
        let workspace = shared.with_shared_model();
        assert!(Arc::ptr_eq(&shared.model, &workspace.model));
        workspace.embed("workspace symbol").unwrap();
        assert_eq!(workspace.cache_size(), 1);
        assert_eq!(shared.cache_size(), 0);

        drop(workspace);

        assert_eq!(Arc::strong_count(&shared.model), 1);
        assert_eq!(shared.cache_size(), 0);
    }

    #[test]
    fn dropping_workspace_keeps_other_workspace_model_available() {
        let shared = shared_engine();
        let first = shared.with_shared_model();
        let second = shared.with_shared_model();
        first.embed_batch(&["first symbol"]).unwrap();
        second.embed_batch(&["second symbol"]).unwrap();
        assert_eq!(first.cache_size(), 1);
        assert_eq!(second.cache_size(), 1);
        assert!(Arc::ptr_eq(&first.model, &second.model));

        drop(first);

        assert_eq!(Arc::strong_count(&shared.model), 2);
        assert_eq!(second.cache_size(), 1);
        assert_eq!(second.embed("another symbol").unwrap(), vec![1.0, 0.0]);
        assert_eq!(second.cache_size(), 2);
        assert_eq!(shared.cache_size(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        // Test cosine similarity calculation
        let a = [1.0_f32, 0.0, 0.0];
        let b = [1.0_f32, 0.0, 0.0];

        // Identical vectors should have similarity 1.0
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let similarity = dot / (norm_a * norm_b);
        assert!((similarity - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_similarity_orthogonal() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];

        // Orthogonal vectors should have similarity 0.0
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_eq!(dot, 0.0);
    }
}
