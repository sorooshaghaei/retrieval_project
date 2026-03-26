from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..cache.store import (
    OBJECT_MEMORY_CACHE,
    _dataframe_fingerprint,
    _hash_payload,
    _load_pickle,
    _normalization_signature,
    _save_pickle,
    ensure_cache_dirs,
)
from ..config import AllConfig, DEFAULT_CONFIG
from ..data.loading import require_columns


def build_or_load_category_classifier(
    train_frame: pd.DataFrame,
    cache_dir: Path,
    config: AllConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `scikit-learn`. Install it with `%pip install scikit-learn`."
        ) from exc

    directories = ensure_cache_dirs(cache_dir)
    require_columns(train_frame, ["id", "content", "category"], "Classifier training data")
    train_signature = _dataframe_fingerprint(train_frame, ["id", "content", "category"])
    normalization_signature = _normalization_signature(config)
    tfidf_params = dict(config.tfidf.__dict__)
    cache_key = _hash_payload(
        {
            "train_signature": train_signature,
            "normalization_signature": normalization_signature,
            "classifier_tfidf_params": tfidf_params,
        }
    )
    cache_path = directories["classifier"] / f"category_classifier_{cache_key}.pkl"
    memory_key = str(cache_path.resolve())
    if memory_key in OBJECT_MEMORY_CACHE:
        return OBJECT_MEMORY_CACHE[memory_key]
    if config.retrieval_pipeline.enable_classic_cache and cache_path.exists():
        print(f"Loading category classifier from cache: {cache_path.name}")
        artifacts = _load_pickle(cache_path)
        OBJECT_MEMORY_CACHE[memory_key] = artifacts
        return artifacts

    vectorizer = TfidfVectorizer(**tfidf_params)
    train_vectors = vectorizer.fit_transform(train_frame["content"])
    classifier = LinearSVC()
    classifier.fit(train_vectors, train_frame["category"].astype(str))
    artifacts = {"vectorizer": vectorizer, "classifier": classifier}
    if config.retrieval_pipeline.enable_classic_cache:
        _save_pickle(cache_path, artifacts)
        print(f"Saved category classifier to cache: {cache_path.name}")
    OBJECT_MEMORY_CACHE[memory_key] = artifacts
    return artifacts


def predict_category_map(query_frame: pd.DataFrame, classifier_artifacts: dict[str, Any]) -> dict[str, str]:
    query_vectors = classifier_artifacts["vectorizer"].transform(query_frame["content"])
    predictions = classifier_artifacts["classifier"].predict(query_vectors)
    return {
        str(query_id): str(prediction)
        for query_id, prediction in zip(query_frame["id"].astype(str), predictions)
    }


def build_doc_category_map(docs_frame: pd.DataFrame) -> dict[str, Any]:
    require_columns(docs_frame, ["id", "category"], "Documents frame")
    return (
        docs_frame[["id", "category"]]
        .assign(id=lambda frame: frame["id"].astype(str))
        .set_index("id")["category"]
        .to_dict()
    )
