from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..cache.store import OBJECT_MEMORY_CACHE, _dataframe_fingerprint, _hash_payload, _load_pickle, _normalization_signature, _save_pickle, ensure_cache_dirs
from ..config import AllConfig, DEFAULT_CONFIG
from ..data.text import tokenize


def _tfidf_param_candidates(config: AllConfig = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    candidates = [dict(config.tfidf.__dict__)]
    min_df = config.tfidf.min_df
    if isinstance(min_df, int) and min_df > 1:
        fallback = dict(config.tfidf.__dict__)
        fallback["min_df"] = 1
        candidates.append(fallback)
    return candidates


def build_or_load_tfidf_index(
    docs_frame: pd.DataFrame,
    cache_dir: Path,
    config: AllConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `scikit-learn`. Install it with `%pip install scikit-learn`."
        ) from exc

    directories = ensure_cache_dirs(cache_dir)
    docs_signature = _dataframe_fingerprint(docs_frame, ["id", "content"])
    normalization_signature = _normalization_signature(config)
    doc_ids = docs_frame["id"].to_numpy()

    for params in _tfidf_param_candidates(config):
        cache_key = _hash_payload(
            {
                "docs_signature": docs_signature,
                "normalization_signature": normalization_signature,
                "tfidf_params": params,
            }
        )
        cache_path = directories["tfidf"] / f"tfidf_{cache_key}.pkl"
        memory_key = str(cache_path.resolve())
        if memory_key in OBJECT_MEMORY_CACHE:
            return OBJECT_MEMORY_CACHE[memory_key]
        if config.retrieval_pipeline.enable_classic_cache and cache_path.exists():
            print(f"Loading TF-IDF artifacts from cache: {cache_path.name}")
            artifacts = _load_pickle(cache_path)
            OBJECT_MEMORY_CACHE[memory_key] = artifacts
            return artifacts

    params = dict(config.tfidf.__dict__)
    vectorizer = TfidfVectorizer(**params)
    try:
        doc_vectors = vectorizer.fit_transform(docs_frame["content"])
    except ValueError as err:
        if "After pruning, no terms remain" not in str(err) or params.get("min_df", 1) == 1:
            raise
        params["min_df"] = 1
        vectorizer = TfidfVectorizer(**params)
        doc_vectors = vectorizer.fit_transform(docs_frame["content"])

    cache_key = _hash_payload(
        {
            "docs_signature": docs_signature,
            "normalization_signature": normalization_signature,
            "tfidf_params": params,
        }
    )
    cache_path = directories["tfidf"] / f"tfidf_{cache_key}.pkl"
    memory_key = str(cache_path.resolve())
    artifacts = {
        "vectorizer": vectorizer,
        "doc_vectors": doc_vectors,
        "doc_ids": doc_ids,
        "params": params,
    }
    if config.retrieval_pipeline.enable_classic_cache:
        _save_pickle(cache_path, artifacts)
        print(f"Saved TF-IDF artifacts to cache: {cache_path.name}")
    OBJECT_MEMORY_CACHE[memory_key] = artifacts
    return artifacts


def build_or_load_bm25_index(
    docs_frame: pd.DataFrame,
    cache_dir: Path,
    config: AllConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    try:
        from rank_bm25 import BM25Plus
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `rank_bm25`. Install it with `%pip install rank-bm25`."
        ) from exc

    directories = ensure_cache_dirs(cache_dir)
    docs_signature = _dataframe_fingerprint(docs_frame, ["id", "content"])
    normalization_signature = _normalization_signature(config)
    cache_key = _hash_payload(
        {
            "docs_signature": docs_signature,
            "normalization_signature": normalization_signature,
            "bm25_params": config.bm25.__dict__,
        }
    )
    cache_path = directories["bm25"] / f"bm25_{cache_key}.pkl"
    memory_key = str(cache_path.resolve())
    if memory_key in OBJECT_MEMORY_CACHE:
        return OBJECT_MEMORY_CACHE[memory_key]
    if config.retrieval_pipeline.enable_classic_cache and cache_path.exists():
        print(f"Loading BM25 index from cache: {cache_path.name}")
        artifacts = _load_pickle(cache_path)
        OBJECT_MEMORY_CACHE[memory_key] = artifacts
        return artifacts

    tokenized_corpus = [tokenize(text) for text in docs_frame["content"]]
    bm25 = BM25Plus(tokenized_corpus, **config.bm25.__dict__)
    artifacts = {"bm25": bm25, "doc_ids": docs_frame["id"].to_numpy()}
    if config.retrieval_pipeline.enable_classic_cache:
        _save_pickle(cache_path, artifacts)
        print(f"Saved BM25 index to cache: {cache_path.name}")
    OBJECT_MEMORY_CACHE[memory_key] = artifacts
    return artifacts
