from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..config import AllConfig, DEFAULT_CONFIG

MODEL_MEMORY_CACHE: dict[str, Any] = {}
ARRAY_MEMORY_CACHE: dict[str, Any] = {}
OBJECT_MEMORY_CACHE: dict[str, Any] = {}


def ensure_cache_dirs(cache_dir: Path) -> dict[str, Path]:
    directories = {
        "root": cache_dir,
        "sentence_transformers": cache_dir / "sentence_transformers",
        "embeddings": cache_dir / "embeddings",
        "tfidf": cache_dir / "tfidf",
        "bm25": cache_dir / "bm25",
        "classifier": cache_dir / "classifier",
        "cross_encoder": cache_dir / "cross_encoder",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _safe_component(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def _hash_payload(payload: dict[str, Any]) -> str:
    raw_payload = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(raw_payload.encode("utf-8")).hexdigest()[:16]


def _normalization_signature(config: AllConfig = DEFAULT_CONFIG) -> str:
    payload = {
        "normalization": config.normalization.__dict__,
        "token_pattern": config.retrieval_pipeline.token_pattern,
    }
    return _hash_payload(payload)


def _dataframe_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    hasher = hashlib.sha1()
    hasher.update(str(len(frame)).encode("utf-8"))
    for column in columns:
        hasher.update(column.encode("utf-8"))
        column_hash = pd.util.hash_pandas_object(frame[column].astype(str), index=False).values
        hasher.update(column_hash.tobytes())
    return hasher.hexdigest()[:16]


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _save_pickle(path: Path, artifact: Any) -> None:
    with open(path, "wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
