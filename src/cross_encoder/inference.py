from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache.store import OBJECT_MEMORY_CACHE, _dataframe_fingerprint, _hash_payload, _safe_component, ensure_cache_dirs
from ..config import AllConfig, DEFAULT_CONFIG
from ..retrieval import prepare_retriever
from ..types import GroundTruthEntry
from .training import build_cross_encoder_training_examples, build_text_map, mine_hard_negative_doc_ids


def _synchronize_cross_encoder_max_length(cross_encoder: Any, max_length: int) -> Any:
    cross_encoder.max_length = int(max_length)
    tokenizer = getattr(cross_encoder, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.model_max_length = int(max_length)
        if hasattr(tokenizer, "init_kwargs"):
            tokenizer.init_kwargs["model_max_length"] = int(max_length)
    tokenizer_kwargs = getattr(cross_encoder, "tokenizer_kwargs", None)
    if isinstance(tokenizer_kwargs, dict):
        tokenizer_kwargs["model_max_length"] = int(max_length)
        tokenizer_kwargs["max_length"] = int(max_length)
    return cross_encoder


def build_or_load_cross_encoder(
    train_queries_frame: pd.DataFrame,
    docs_frame: pd.DataFrame,
    ground_truth: dict[str, GroundTruthEntry],
    cache_dir: Path,
    config: AllConfig = DEFAULT_CONFIG,
) -> Any:
    try:
        from sentence_transformers import CrossEncoder
        from torch.utils.data import DataLoader
        import torch
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies for cross-encoder training. Install `%pip install sentence-transformers torch`."
        ) from exc

    directories = ensure_cache_dirs(cache_dir)
    query_text_map = build_text_map(train_queries_frame, id_column="id", text_column="content")
    doc_text_map = build_text_map(docs_frame, id_column="id", text_column="content")
    candidate_query_ids = [
        query_id
        for query_id in train_queries_frame["id"].astype(str).tolist()
        if query_id in ground_truth and query_id in query_text_map
    ]
    if config.cross_encoder.train_query_limit > 0:
        candidate_query_ids = candidate_query_ids[: config.cross_encoder.train_query_limit]
    if not candidate_query_ids:
        raise ValueError("No training queries available for cross-encoder training.")
    train_signature = _hash_payload(
        {
            "query_signature": _dataframe_fingerprint(train_queries_frame, ["id", "content"]),
            "doc_signature": _dataframe_fingerprint(docs_frame, ["id", "content"]),
            "query_count": len(candidate_query_ids),
            "model": config.cross_encoder.model_name,
            "epochs": config.cross_encoder.epochs,
            "batch_size": config.cross_encoder.batch_size,
            "max_length": config.cross_encoder.max_length,
            "max_pos_per_query": config.cross_encoder.max_positives_per_query,
            "neg_per_pos": config.cross_encoder.negatives_per_positive,
            "hard_neg_top_k": config.cross_encoder.hard_negative_top_k,
            "hard_neg_model": "embedding",
            "hard_neg_embedding_model": config.embedding.model_name,
            "seed": config.cross_encoder.random_seed,
        }
    )
    model_dir = directories["cross_encoder"] / f"{_safe_component(config.cross_encoder.model_name)}_{train_signature}"
    cache_marker = model_dir / "config.json"
    memory_key = f"cross_encoder::{model_dir.resolve()}"
    if memory_key in OBJECT_MEMORY_CACHE:
        return OBJECT_MEMORY_CACHE[memory_key]
    if config.cross_encoder.enable_cache and cache_marker.exists():
        print(f"Loading cross-encoder from cache: {model_dir.name}")
        try:
            cached_model = _synchronize_cross_encoder_max_length(
                CrossEncoder(str(model_dir), max_length=config.cross_encoder.max_length),
                config.cross_encoder.max_length,
            )
            if config.cross_encoder.fp16 and torch.cuda.is_available():
                cached_model.model.half()
            OBJECT_MEMORY_CACHE[memory_key] = cached_model
            return cached_model
        except Exception as exc:
            print(f"Cross-encoder cache load failed, retraining: {exc}")
    elif config.cross_encoder.enable_cache and model_dir.exists():
        print(f"Cross-encoder cache directory exists but is incomplete: {model_dir}")

    embedding_artifacts = prepare_retriever("embedding", docs_frame, cache_dir=cache_dir, config=config)
    hard_negative_doc_ids_by_query = mine_hard_negative_doc_ids(
        train_queries_frame=train_queries_frame,
        docs_frame=docs_frame,
        ground_truth=ground_truth,
        query_ids=candidate_query_ids,
        top_k=config.cross_encoder.hard_negative_top_k,
        prepared_artifacts=embedding_artifacts,
        cache_dir=cache_dir,
        config=config,
    )
    training_examples = build_cross_encoder_training_examples(
        query_text_map=query_text_map,
        doc_text_map=doc_text_map,
        ground_truth=ground_truth,
        query_ids=candidate_query_ids,
        max_positives_per_query=config.cross_encoder.max_positives_per_query,
        negatives_per_positive=config.cross_encoder.negatives_per_positive,
        hard_negative_doc_ids_by_query=hard_negative_doc_ids_by_query,
        seed=config.cross_encoder.random_seed,
        config=config,
    )
    if not training_examples:
        raise ValueError("Cross-encoder training set is empty after preprocessing.")
    print(f"Training cross-encoder on {len(training_examples):,} pairs from {len(candidate_query_ids):,} queries")
    cross_encoder = _synchronize_cross_encoder_max_length(
        CrossEncoder(config.cross_encoder.model_name, max_length=config.cross_encoder.max_length),
        config.cross_encoder.max_length,
    )
    train_loader = DataLoader(training_examples, shuffle=True, batch_size=config.cross_encoder.batch_size)
    warmup_steps = max(1, int(len(train_loader) * config.cross_encoder.epochs * 0.1))
    if config.cross_encoder.enable_cache:
        model_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(model_dir) if config.cross_encoder.enable_cache else None
    cross_encoder.fit(
        train_dataloader=train_loader,
        epochs=config.cross_encoder.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=output_path,
    )
    if config.cross_encoder.enable_cache:
        print(f"Saved cross-encoder to cache: {model_dir.name}")
        cross_encoder.save(str(model_dir))
        cached_model = _synchronize_cross_encoder_max_length(
            CrossEncoder(str(model_dir), max_length=config.cross_encoder.max_length),
            config.cross_encoder.max_length,
        )
        if config.cross_encoder.fp16 and torch.cuda.is_available():
            cached_model.model.half()
        OBJECT_MEMORY_CACHE[memory_key] = cached_model
        return cached_model
    if config.cross_encoder.fp16 and torch.cuda.is_available():
        cross_encoder.model.half()
    OBJECT_MEMORY_CACHE[memory_key] = cross_encoder
    return cross_encoder


def evaluate_isolated_cross_encoder(
    cross_encoder: Any,
    query_frame: pd.DataFrame,
    docs_frame: pd.DataFrame,
    ground_truth: dict[str, dict],
    hard_negative_doc_ids_by_query: dict[str, list[str]],
    sample_limit: int = 100,
) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    print("Running isolated Cross-Encoder diagnostics...")
    query_text_map = build_text_map(query_frame, id_column="id", text_column="content")
    doc_text_map = build_text_map(docs_frame, id_column="id", text_column="content")
    model_inputs = []
    labels = []
    queries_tested = 0
    for query_id, info in ground_truth.items():
        if queries_tested >= sample_limit:
            break
        query_id_str = str(query_id)
        if query_id_str not in query_text_map:
            continue
        query_text = query_text_map[query_id_str]
        relevant_doc_ids = [str(doc_id) for doc_id in info.get("relevant_doc_ids", []) if str(doc_id) in doc_text_map]
        if not relevant_doc_ids:
            continue
        hard_negatives = hard_negative_doc_ids_by_query.get(query_id_str, [])
        hard_negatives = [doc_id for doc_id in hard_negatives if doc_id in doc_text_map and doc_id not in relevant_doc_ids][:10]
        if not hard_negatives:
            continue
        for doc_id in relevant_doc_ids:
            model_inputs.append([query_text, doc_text_map[doc_id]])
            labels.append(1.0)
        for doc_id in hard_negatives:
            model_inputs.append([query_text, doc_text_map[doc_id]])
            labels.append(0.0)
        queries_tested += 1
    if not labels:
        print("Not enough valid pairs to run diagnostics.")
        return
    print(f"Scoring {len(labels)} pairs ({sum(labels)} Positives, {len(labels) - sum(labels)} Negatives)...")
    scores = np.asarray(cross_encoder.predict(model_inputs, batch_size=32, show_progress_bar=True), dtype=np.float32).reshape(-1)
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    print("\n=== Isolated Cross-Encoder Metrics ===")
    print(f"ROC-AUC:           {roc_auc:.4f} (1.0 is perfect separation, 0.5 is random guessing)")
    print(f"Average Precision: {pr_auc:.4f} (Higher is better, measures precision-recall curve)")
    pos_scores = scores[np.array(labels) == 1.0]
    neg_scores = scores[np.array(labels) == 0.0]
    print(f"\nMean Score (Positives): {np.mean(pos_scores):.4f}")
    print(f"Mean Score (Negatives): {np.mean(neg_scores):.4f}")
