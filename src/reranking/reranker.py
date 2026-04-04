from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..cross_encoder.training import build_text_map
from ..types import RetrievalResult


def compute_category_boost(
    query_id: str,
    doc_ids: list[str],
    query_category_map: dict[str, str],
    doc_category_map: dict[str, Any],
    category_bonus: float,
) -> np.ndarray:
    boosts = np.zeros(len(doc_ids), dtype=np.float32)
    predicted_query_cat = query_category_map.get(query_id)
    if predicted_query_cat is None or category_bonus == 0.0:
        return boosts
    target_category = str(predicted_query_cat)
    for index, doc_id in enumerate(doc_ids):
        raw_doc_cat = doc_category_map.get(doc_id)
        doc_category = "unknown" if raw_doc_cat is None or pd.isna(raw_doc_cat) else str(raw_doc_cat)
        if doc_category == target_category:
            boosts[index] = category_bonus
    return boosts


def rerank_results_with_cross_encoder(
    results: list[RetrievalResult],
    query_frame: pd.DataFrame,
    docs_frame: pd.DataFrame,
    cross_encoder: Any,
    query_category_map: dict[str, str],
    doc_category_map: dict[str, Any],
    infer_batch_size: int,
    rerank_top_m: int = 100,
    category_bonus: float = 2.0,
    return_diagnostics: bool = False,
) -> list[RetrievalResult] | tuple[list[RetrievalResult], list[dict[str, Any]]]:
    if rerank_top_m <= 0:
        raise ValueError("rerank_top_m must be positive.")
    query_text_map = build_text_map(query_frame, id_column="id", text_column="content")
    doc_text_map = build_text_map(docs_frame, id_column="id", text_column="content")
    all_pairs: list[list[str]] = []
    query_meta: list[dict[str, Any]] = []
    for item in results:
        query_id = str(item["query_id"])
        doc_ids = [str(doc_id) for doc_id in item["relevant_docs"]]
        if query_id not in query_text_map:
            query_meta.append(
                {"query_id": query_id, "skip": True, "skip_reason": "missing_query_text", "doc_ids": doc_ids, "head_doc_ids": doc_ids[:rerank_top_m]}
            )
            continue
        head_doc_ids = doc_ids[:rerank_top_m]
        tail_doc_ids = doc_ids[rerank_top_m:]
        scored_doc_ids = [doc_id for doc_id in head_doc_ids if doc_id in doc_text_map]
        missing_head_doc_ids = [doc_id for doc_id in head_doc_ids if doc_id not in doc_text_map]
        if len(scored_doc_ids) <= 1:
            query_meta.append(
                {"query_id": query_id, "skip": True, "skip_reason": "insufficient_scored_docs", "doc_ids": doc_ids, "head_doc_ids": head_doc_ids}
            )
            continue
        pairs = [[query_text_map[query_id], doc_text_map[doc_id]] for doc_id in scored_doc_ids]
        start = len(all_pairs)
        all_pairs.extend(pairs)
        query_meta.append(
            {
                "query_id": query_id,
                "skip": False,
                "head_doc_ids": head_doc_ids,
                "scored_doc_ids": scored_doc_ids,
                "missing_head_doc_ids": missing_head_doc_ids,
                "tail_doc_ids": tail_doc_ids,
                "slice": (start, start + len(pairs)),
            }
        )
    all_scores = (
        np.asarray(cross_encoder.predict(all_pairs, batch_size=infer_batch_size, show_progress_bar=False), dtype=np.float32)
        if all_pairs
        else np.array([], dtype=np.float32)
    )
    reranked_results: list[RetrievalResult] = []
    rerank_diagnostics: list[dict[str, Any]] = []
    reranked_query_count = 0
    for meta in query_meta:
        if meta["skip"]:
            reranked_results.append({"query_id": meta["query_id"], "relevant_docs": meta["doc_ids"]})
            if return_diagnostics:
                rerank_diagnostics.append(
                    {
                        "query_id": meta["query_id"],
                        "reranked": False,
                        "skip_reason": meta["skip_reason"],
                        "rerank_top_m": int(rerank_top_m),
                        "category_bonus": float(category_bonus),
                        "original_head_doc_ids": meta.get("head_doc_ids", []),
                        "reranked_head_doc_ids": meta.get("head_doc_ids", []),
                        "missing_head_doc_ids": [],
                        "tail_doc_ids_count": max(0, len(meta["doc_ids"]) - len(meta.get("head_doc_ids", []))),
                        "candidate_docs": [],
                    }
                )
            continue
        start, end = meta["slice"]
        ce_scores = all_scores[start:end]
        scored_doc_ids = meta["scored_doc_ids"]
        boosts = compute_category_boost(meta["query_id"], scored_doc_ids, query_category_map, doc_category_map, category_bonus)
        final_scores = ce_scores + boosts
        ranked_indices = np.argsort(final_scores)[::-1]
        reranked_scored_doc_ids = [scored_doc_ids[index] for index in ranked_indices]
        reranked_doc_ids = reranked_scored_doc_ids + meta["missing_head_doc_ids"] + meta["tail_doc_ids"]
        reranked_results.append({"query_id": meta["query_id"], "relevant_docs": reranked_doc_ids})
        reranked_query_count += 1
        if return_diagnostics:
            reranked_positions = {doc_id: rank for rank, doc_id in enumerate(reranked_scored_doc_ids, start=1)}
            candidate_docs: list[dict[str, Any]] = []
            for original_rank, doc_id in enumerate(scored_doc_ids, start=1):
                raw_doc_category = doc_category_map.get(doc_id)
                doc_category = "unknown" if raw_doc_category is None or pd.isna(raw_doc_category) else str(raw_doc_category)
                score_index = original_rank - 1
                candidate_docs.append(
                    {
                        "doc_id": doc_id,
                        "original_rank": int(original_rank),
                        "reranked_rank": int(reranked_positions[doc_id]),
                        "cross_encoder_score": float(ce_scores[score_index]),
                        "category_boost": float(boosts[score_index]),
                        "final_score": float(final_scores[score_index]),
                        "doc_category": doc_category,
                    }
                )
            rerank_diagnostics.append(
                {
                    "query_id": meta["query_id"],
                    "reranked": True,
                    "skip_reason": None,
                    "rerank_top_m": int(rerank_top_m),
                    "category_bonus": float(category_bonus),
                    "predicted_query_category": query_category_map.get(meta["query_id"]),
                    "original_head_doc_ids": meta["head_doc_ids"],
                    "reranked_head_doc_ids": reranked_scored_doc_ids + meta["missing_head_doc_ids"],
                    "missing_head_doc_ids": meta["missing_head_doc_ids"],
                    "tail_doc_ids_count": len(meta["tail_doc_ids"]),
                    "candidate_docs": candidate_docs,
                }
            )
    print(
        f"  [CrossEncoder] reranked {reranked_query_count:,}/{len(results):,} queries "
        f"(top_m={rerank_top_m:,}, category_bonus={category_bonus:.2f}, total_pairs={len(all_pairs):,})"
    )
    if return_diagnostics:
        return reranked_results, rerank_diagnostics
    return reranked_results
