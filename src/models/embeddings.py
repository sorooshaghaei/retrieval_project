from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache.store import (
    ARRAY_MEMORY_CACHE,
    MODEL_MEMORY_CACHE,
    _dataframe_fingerprint,
    _normalization_signature,
    _safe_component,
    ensure_cache_dirs,
)
from ..config import AllConfig, DEFAULT_CONFIG


def _load_sentence_model(model_name: str, cache_dir: Path, config: AllConfig = DEFAULT_CONFIG) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `sentence_transformers`. Install it with `%pip install sentence-transformers`."
        ) from exc

    if model_name in MODEL_MEMORY_CACHE:
        return MODEL_MEMORY_CACHE[model_name]

    model_cache_dir = ensure_cache_dirs(cache_dir)["sentence_transformers"]
    safe_model_name = _safe_component(model_name)
    local_model_dir = model_cache_dir / safe_model_name
    if local_model_dir.exists():
        print(f"Loading model weights from cache: {local_model_dir}")
        model = SentenceTransformer(str(local_model_dir))
    else:
        print(f"Downloading model weights: {model_name}")
        model = SentenceTransformer(model_name)
        local_model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(local_model_dir))
        print(f"Saved model weights to cache: {local_model_dir}")

    MODEL_MEMORY_CACHE[model_name] = model
    return model


def _load_or_encode_embeddings(
    frame: pd.DataFrame,
    kind: str,
    model: Any,
    model_name: str,
    batch_size: int,
    cache_dir: Path,
    config: AllConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    directories = ensure_cache_dirs(cache_dir)
    signature = _dataframe_fingerprint(frame, ["id", "content"])
    normalization_signature = _normalization_signature(config)
    cache_name = f"{kind}_{_safe_component(model_name)}_{normalization_signature}_{signature}.npy"
    cache_path = directories["embeddings"] / cache_name
    memory_key = str(cache_path.resolve())

    if memory_key in ARRAY_MEMORY_CACHE:
        return ARRAY_MEMORY_CACHE[memory_key]

    if config.retrieval_pipeline.enable_embedding_cache and cache_path.exists():
        print(f"Loading {kind} embeddings from cache: {cache_path.name}")
        embeddings = np.load(cache_path)
    else:
        print(f"Encoding {len(frame):,} {kind} rows...")
        embeddings = model.encode(
            frame["content"].tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if config.retrieval_pipeline.enable_embedding_cache:
            np.save(cache_path, embeddings)
            print(f"Saved {kind} embeddings to cache: {cache_path.name}")

    ARRAY_MEMORY_CACHE[memory_key] = embeddings
    return embeddings
