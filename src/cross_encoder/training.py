from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import AllConfig, DEFAULT_CONFIG
from ..data.loading import require_columns
from ..retrieval import prepare_retriever, run_retrieval
from ..types import GroundTruthEntry


def build_text_map(frame: pd.DataFrame, id_column: str = "id", text_column: str = "content") -> dict[str, str]:
    require_columns(frame, [id_column, text_column], f"Frame[{id_column}, {text_column}]")
    return {
        str(row_id): str(text)
        for row_id, text in frame[[id_column, text_column]].itertuples(index=False, name=None)
    }


def sample_random_negative_doc_ids(
    all_doc_ids: Sequence[str],
    excluded_doc_ids: set[str],
    sample_size: int,
    rng: Any,
) -> list[str]:
    if sample_size <= 0 or not all_doc_ids:
        return []
    negatives: list[str] = []
    max_attempts = max(100, sample_size * 50)
    attempts = 0
    while len(negatives) < sample_size and attempts < max_attempts:
        candidate = str(all_doc_ids[rng.randrange(len(all_doc_ids))])
        attempts += 1
        if candidate in excluded_doc_ids or candidate in negatives:
            continue
        negatives.append(candidate)
    return negatives


def mine_hard_negative_doc_ids(
    train_queries_frame: pd.DataFrame,
    docs_frame: pd.DataFrame,
    ground_truth: dict[str, GroundTruthEntry],
    query_ids: Sequence[str],
    cache_dir: Path,
    top_k: int | None = None,
    prepared_artifacts: dict[str, Any] | None = None,
    config: AllConfig = DEFAULT_CONFIG,
) -> dict[str, list[str]]:
    hard_negative_top_k = top_k or config.cross_encoder.hard_negative_top_k
    if hard_negative_top_k <= 0:
        return {}
    query_id_set = {str(query_id) for query_id in query_ids}
    mining_queries_frame = (
        train_queries_frame.assign(id=train_queries_frame["id"].astype(str))
        .loc[lambda frame: frame["id"].isin(query_id_set), ["id", "content"]]
        .reset_index(drop=True)
    )
    if mining_queries_frame.empty:
        return {}
    embedding_artifacts = prepared_artifacts or prepare_retriever("embedding", docs_frame, cache_dir=cache_dir, config=config)
    retrieval_results = run_retrieval(
        model_name="embedding",
        docs_frame=docs_frame,
        queries_frame=mining_queries_frame,
        top_k=hard_negative_top_k,
        cache_dir=cache_dir,
        prepared_artifacts=embedding_artifacts,
        embedding_kind="queries_train_hardneg",
        config=config,
    )
    hard_negative_doc_ids_by_query: dict[str, list[str]] = {}
    for item in retrieval_results:
        query_id = str(item["query_id"])
        relevant_doc_ids = ground_truth[query_id]["relevant_doc_ids"] if query_id in ground_truth else set()
        negatives: list[str] = []
        seen_doc_ids: set[str] = set()
        for doc_id in item["relevant_docs"]:
            candidate_doc_id = str(doc_id)
            if candidate_doc_id in relevant_doc_ids or candidate_doc_id in seen_doc_ids:
                continue
            negatives.append(candidate_doc_id)
            seen_doc_ids.add(candidate_doc_id)
        hard_negative_doc_ids_by_query[query_id] = negatives
    counts = [len(doc_ids) for doc_ids in hard_negative_doc_ids_by_query.values()]
    if counts:
        print(
            f"  [HardNegatives] mined for {len(counts):,} queries "
            f"(per-query min/mean/max={min(counts):,}/{float(np.mean(counts)):.1f}/{max(counts):,}, top_k={hard_negative_top_k:,})"
        )
    return hard_negative_doc_ids_by_query


def build_cross_encoder_training_examples(
    query_text_map: dict[str, str],
    doc_text_map: dict[str, str],
    ground_truth: dict[str, GroundTruthEntry],
    query_ids: Sequence[str],
    max_positives_per_query: int | None = None,
    negatives_per_positive: int | None = None,
    hard_negative_doc_ids_by_query: dict[str, list[str]] | None = None,
    seed: int | None = None,
    config: AllConfig = DEFAULT_CONFIG,
) -> list[Any]:
    try:
        from sentence_transformers import InputExample
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `sentence_transformers`. Install it with `%pip install sentence-transformers`."
        ) from exc

    import random

    max_positives = max_positives_per_query or config.cross_encoder.max_positives_per_query
    negatives = negatives_per_positive or config.cross_encoder.negatives_per_positive
    rng = random.Random(config.cross_encoder.random_seed if seed is None else seed)
    all_doc_ids = list(doc_text_map.keys())
    examples: list[Any] = []
    hard_negative_count = 0
    random_negative_count = 0
    for query_id in query_ids:
        query_id_str = str(query_id)
        query_text = query_text_map.get(query_id_str)
        if query_text is None or query_id_str not in ground_truth:
            continue
        relevant_doc_ids = [
            str(doc_id)
            for doc_id in ground_truth[query_id_str]["relevant_doc_ids"]
            if str(doc_id) in doc_text_map
        ]
        if not relevant_doc_ids:
            continue
        rng.shuffle(relevant_doc_ids)
        selected_positive_doc_ids = relevant_doc_ids[:max_positives]
        relevant_doc_id_set = set(relevant_doc_ids)
        hard_negative_pool = [
            doc_id
            for doc_id in (hard_negative_doc_ids_by_query or {}).get(query_id_str, [])
            if doc_id in doc_text_map and doc_id not in relevant_doc_id_set
        ]
        for positive_doc_id in selected_positive_doc_ids:
            examples.append(InputExample(texts=[query_text, doc_text_map[positive_doc_id]], label=1.0))
            negative_doc_ids: list[str] = []
            if hard_negative_pool:
                hard_take = min(negatives, len(hard_negative_pool))
                negative_doc_ids.extend(rng.sample(hard_negative_pool, hard_take))
                hard_negative_count += hard_take
            if len(negative_doc_ids) < negatives:
                random_needed = negatives - len(negative_doc_ids)
                extra_random_negatives = sample_random_negative_doc_ids(
                    all_doc_ids=all_doc_ids,
                    excluded_doc_ids=relevant_doc_id_set | set(negative_doc_ids),
                    sample_size=random_needed,
                    rng=rng,
                )
                negative_doc_ids.extend(extra_random_negatives)
                random_negative_count += len(extra_random_negatives)
            for negative_doc_id in negative_doc_ids:
                examples.append(InputExample(texts=[query_text, doc_text_map[negative_doc_id]], label=0.0))
    print(
        f"Cross-encoder pairs: total={len(examples):,}, "
        f"hard_negatives={hard_negative_count:,}, random_negatives={random_negative_count:,}"
    )
    return examples
