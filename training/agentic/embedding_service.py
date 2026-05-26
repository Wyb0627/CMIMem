"""Shared embedding service for agentic RL.

Provides a Ray Named Actor that holds a SentenceTransformer model on GPU.
All AgentLoopWorker processes call this actor via Ray RPC, avoiding redundant
model loading across 16+ workers.

NOTE on GPU memory safety:
  In verl's hybrid mode (free_cache_engine=True), GPU memory is shared between
  FSDP training and vLLM inference via sleep/wake_up cycles. The embedding model
  MUST be small enough (~1-2GB) so that its permanent VRAM footprint does not
  prevent vLLM from successfully waking up. A 0.6B model (~1.2GB) is safe;
  a 4B model (~8GB) will deadlock.

Usage:
    from training.agentic.embedding_service import get_embedding_actor

    actor = get_embedding_actor(model_path)
    vectors = ray.get(actor.encode.remote(texts, batch_size=64))
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import ray

logger = logging.getLogger(__name__)

EMBEDDING_ACTOR_NAME = "shared_embedding_service"

# Max model VRAM footprint (GB) that is safe to co-exist with vLLM wake_up.
# 80GB GPU with gpu_memory_utilization=0.7 leaves ~14GB headroom after
# vLLM model weights + KV cache. Keep well below that.
_MAX_SAFE_MODEL_VRAM_GB = 4.0


@ray.remote(num_gpus=0, num_cpus=2)
class EmbeddingServiceActor:
    """Ray actor holding a SentenceTransformer model on CPU.

    Uses num_gpus=0 so Ray can always schedule this actor regardless of GPU
    availability. With the lightweight 0.6B embedding model, CPU inference
    is fast enough (~50ms per batch) and avoids GPU scheduling conflicts
    in both batch and agentic training modes.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(
            f"[EmbeddingService] Loading {model_path} on {device} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})...",
            flush=True,
        )
        self.model = SentenceTransformer(
            model_path, device=device, trust_remote_code=True
        )
        self.device = device

        try:
            dim = self.model.get_sentence_embedding_dimension()
        except AttributeError:
            dim = self.model.get_embedding_dimension()

        # Log VRAM usage if on GPU
        vram_info = ""
        if device.startswith("cuda") and torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            vram_info = f", VRAM={allocated_gb:.2f}GB"
            if allocated_gb > _MAX_SAFE_MODEL_VRAM_GB:
                print(
                    f"[EmbeddingService] WARNING: Model uses {allocated_gb:.1f}GB VRAM, "
                    f"exceeding safe limit ({_MAX_SAFE_MODEL_VRAM_GB}GB). "
                    f"This may cause vLLM wake_up to OOM. Consider using a smaller model.",
                    flush=True,
                )

        print(
            f"[EmbeddingService] Model loaded. dim={dim}, device={device}{vram_info}",
            flush=True,
        )

    def encode(
        self, texts: List[str], batch_size: int = 64
    ) -> List[List[float]]:
        """Encode texts and return as nested list (serializable via Ray)."""
        import numpy as np

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        try:
            return self.model.get_sentence_embedding_dimension()
        except AttributeError:
            return self.model.get_embedding_dimension()


def get_embedding_actor(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> ray.actor.ActorHandle:
    """Get or create the shared embedding service actor.

    Idempotent: if the named actor already exists, returns the existing handle.
    Otherwise creates a new one with the given model_path.

    Args:
        model_path: Path to the SentenceTransformer model. Required on first call.
        device: Device string (e.g. "cuda"). Defaults to auto-detect.

    Returns:
        Ray actor handle for EmbeddingServiceActor.
    """
    try:
        return ray.get_actor(EMBEDDING_ACTOR_NAME)
    except ValueError:
        pass

    if not model_path:
        raise ValueError(
            "Embedding service actor does not exist yet. "
            "Provide model_path to create it."
        )

    # Resolve model path
    from llm_client import _resolve_local_model_path

    local_path = _resolve_local_model_path(model_path, auto_download=True)

    actor = EmbeddingServiceActor.options(
        name=EMBEDDING_ACTOR_NAME,
        lifetime="detached",
        get_if_exists=True,
    ).remote(local_path, device)

    # Wait for initialization with timeout to avoid deadlock when all GPUs
    # are occupied by vLLM workers and the actor cannot be scheduled.
    ray.get(actor.get_dimension.remote(), timeout=30)
    print(f"[EmbeddingService] Actor '{EMBEDDING_ACTOR_NAME}' ready.", flush=True)
    return actor