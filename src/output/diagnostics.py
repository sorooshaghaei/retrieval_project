from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ..types import GroundTruthEntry, RetrievalResult


def to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    if isinstance(value, set):
        return [to_json_ready(item) for item in sorted(value)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def preview_text(value: Any, limit: int = 280) -> str:
    text = "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def format_category_percentages(doc_ids: list[str], doc_categories: dict[str, Any]) -> str:
    if not doc_ids:
        return ""
    categories = []
    for doc_id in doc_ids:
        raw_category = doc_categories.get(str(doc_id))
        category = "unknown" if raw_category is None or pd.isna(raw_category) else str(raw_category)
        categories.append(category)
    counts = Counter(categories)
    total = len(doc_ids)
    return ", ".join(f"{category}: {count / total * 100:.1f}%" for category, count in counts.most_common())


def top_category_percentages_by_query(
    results: list[RetrievalResult],
    doc_categories: dict[str, Any],
    top_n: int = 20,
) -> dict[str, str]:
    return {
        str(item["query_id"]): format_category_percentages(item["relevant_docs"][:top_n], doc_categories)
        for item in results
    }


def build_scalar_map(frame: pd.DataFrame, value_column: str) -> dict[str, Any]:
    return frame.assign(id=frame["id"].astype(str)).set_index("id")[value_column].to_dict()


def compute_query_metrics(
    result: RetrievalResult,
    ground_truth: dict[str, GroundTruthEntry],
    k: int,
) -> dict[str, Any]:
    query_id = str(result["query_id"])
    relevant_doc_ids = set(ground_truth.get(query_id, {}).get("relevant_doc_ids", set()))
    retrieved_doc_ids = [str(doc_id) for doc_id in result["relevant_docs"][:k]]
    hits = [doc_id for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids]
    first_relevant_rank = next(
        (rank for rank, doc_id in enumerate(retrieved_doc_ids, start=1) if doc_id in relevant_doc_ids),
        None,
    )
    total_relevant_docs = int(ground_truth.get(query_id, {}).get("total_relevant_docs", len(relevant_doc_ids)))
    return {
        "query_id": query_id,
        "relevant_doc_ids": sorted(relevant_doc_ids),
        "retrieved_doc_ids": retrieved_doc_ids,
        "hit_count": len(hits),
        "recall_at_k": len(hits) / total_relevant_docs if total_relevant_docs > 0 else 0.0,
        "precision_at_k": len(hits) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
    }


def build_relevant_rank_changes(
    base_doc_ids: list[str],
    reranked_doc_ids: list[str],
    relevant_doc_ids: list[str] | set[str],
) -> list[dict[str, Any]]:
    relevant_set = {str(doc_id) for doc_id in relevant_doc_ids}
    base_positions = {str(doc_id): rank for rank, doc_id in enumerate(base_doc_ids, start=1)}
    reranked_positions = {str(doc_id): rank for rank, doc_id in enumerate(reranked_doc_ids, start=1)}
    return [
        {
            "doc_id": doc_id,
            "base_rank": base_positions.get(doc_id),
            "reranked_rank": reranked_positions.get(doc_id),
            "delta": None
            if base_positions.get(doc_id) is None or reranked_positions.get(doc_id) is None
            else base_positions[doc_id] - reranked_positions[doc_id],
        }
        for doc_id in sorted(relevant_set)
        if base_positions.get(doc_id) is not None or reranked_positions.get(doc_id) is not None
    ]


def build_doc_snapshot(doc_id: str, rank: int | None, relevant_doc_ids: set[str]) -> dict[str, Any]:
    return {"doc_id": str(doc_id), "rank": rank, "is_relevant": str(doc_id) in relevant_doc_ids}


def build_ranked_doc_debug_row(
    doc_id: str,
    relevant_doc_ids: set[str],
    candidate_doc_map: dict[str, Any],
    predicted_query_category: str | None,
    base_rank: int | None,
    reranked_rank: int | None,
) -> dict[str, Any]:
    candidate_info = candidate_doc_map.get(str(doc_id), {})
    doc_category = candidate_info.get("doc_category")
    category_match = None if predicted_query_category is None or doc_category is None else str(predicted_query_category) == str(doc_category)
    return {
        "doc_id": str(doc_id),
        "title": candidate_info.get("title"),
        "base_rank": base_rank,
        "reranked_rank": reranked_rank,
        "rank_delta": None if base_rank is None or reranked_rank is None else base_rank - reranked_rank,
        "is_relevant": str(doc_id) in relevant_doc_ids,
        "cross_encoder_score": candidate_info.get("cross_encoder_score"),
        "category_boost": candidate_info.get("category_boost"),
        "final_score": candidate_info.get("final_score"),
        "doc_category": doc_category,
        "predicted_query_category": predicted_query_category,
        "category_match": category_match,
    }


def summarize_reranker_failure(
    query_detail: dict[str, Any],
    demoted_relevant_rows: list[dict[str, Any]],
    promoted_non_relevant_rows: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    base_first_rank = query_detail.get("base_first_relevant_rank")
    reranked_first_rank = query_detail.get("first_relevant_rank")
    if base_first_rank is not None and reranked_first_rank is None:
        reasons.append("reranker removed the first relevant document from the ranking")
    elif base_first_rank is not None and reranked_first_rank is not None and reranked_first_rank > base_first_rank:
        reasons.append(f"first relevant moved from rank {base_first_rank} to rank {reranked_first_rank}")
    if promoted_non_relevant_rows:
        top_non_relevant = promoted_non_relevant_rows[0]
        reasons.append(f"non-relevant document promoted: '{top_non_relevant.get('title') or top_non_relevant['doc_id']}'")
    if demoted_relevant_rows:
        top_relevant = demoted_relevant_rows[0]
        reasons.append(f"relevant document demoted: '{top_relevant.get('title') or top_relevant['doc_id']}'")
    if not reasons:
        reasons.append("reranker regression detected, but no simple summary was extracted")
    return reasons


def build_bad_case_entry(
    query_detail: dict[str, Any],
    query_title_map: dict[str, Any],
    query_text_map: dict[str, Any],
    doc_category_map: dict[str, Any],
    snapshot_limit: int = 10,
) -> dict[str, Any]:
    query_id = str(query_detail["query_id"])
    relevant_doc_id_set = set(query_detail["relevant_doc_ids"])
    predicted_query_category = query_detail.get("predicted_category")
    rerank_diagnostics = query_detail.get("rerank_diagnostics") or {}
    candidate_doc_map = {str(item["doc_id"]): item for item in rerank_diagnostics.get("candidate_docs", [])}
    base_positions = {str(doc_id): rank for rank, doc_id in enumerate(query_detail["base_retrieved_doc_ids"], start=1)}
    reranked_positions = {str(doc_id): rank for rank, doc_id in enumerate(query_detail["retrieved_doc_ids"], start=1)}
    relevant_docs = [
        build_ranked_doc_debug_row(
            doc_id,
            relevant_doc_id_set,
            candidate_doc_map,
            predicted_query_category,
            base_positions.get(str(doc_id)),
            reranked_positions.get(str(doc_id)),
        )
        for doc_id in query_detail["relevant_doc_ids"]
    ]
    demoted_relevant_rows = sorted(
        [
            row
            for row in relevant_docs
            if row["base_rank"] is not None and (row["reranked_rank"] is None or row["reranked_rank"] > row["base_rank"])
        ],
        key=lambda row: (
            0 if row["reranked_rank"] is None else 1,
            -10**9 if row["reranked_rank"] is None else -(row["reranked_rank"] - row["base_rank"]),
            row["base_rank"] or 10**9,
        ),
    )
    promoted_non_relevant_rows = sorted(
        [
            build_ranked_doc_debug_row(
                doc_id,
                relevant_doc_id_set,
                candidate_doc_map,
                predicted_query_category,
                base_positions.get(str(doc_id)),
                reranked_positions.get(str(doc_id)),
            )
            for doc_id, candidate_info in candidate_doc_map.items()
            if doc_id not in relevant_doc_id_set
            and candidate_info.get("original_rank") is not None
            and candidate_info.get("reranked_rank") is not None
            and int(candidate_info["reranked_rank"]) < int(candidate_info["original_rank"])
        ],
        key=lambda row: (
            10**9 if row["base_rank"] is None or row["reranked_rank"] is None else -(row["base_rank"] - row["reranked_rank"]),
            row["reranked_rank"] or 10**9,
        ),
    )
    failure_reasons = summarize_reranker_failure(query_detail, demoted_relevant_rows, promoted_non_relevant_rows)
    return {
        "query": {
            "query_id": query_id,
            "title": query_title_map.get(query_id),
            "text_preview": preview_text(query_text_map.get(query_id)),
            "true_category": query_detail["true_category"],
            "predicted_category": query_detail["predicted_category"],
            "category_correct": query_detail["category_correct"],
        },
        "failure": {
            "delta_hit_count": query_detail["delta_hit_count"],
            "delta_recall_at_k": query_detail["delta_recall_at_k"],
            "delta_precision_at_k": query_detail["delta_precision_at_k"],
            "delta_reciprocal_rank": query_detail["delta_reciprocal_rank"],
            "base_first_relevant_rank": query_detail["base_first_relevant_rank"],
            "reranked_first_relevant_rank": query_detail["first_relevant_rank"],
            "base_hit_count": query_detail["base_hit_count"],
            "reranked_hit_count": query_detail["hit_count"],
            "failure_reasons": failure_reasons,
        },
        "reranker_context": {
            "reranked": rerank_diagnostics.get("reranked"),
            "rerank_top_m": rerank_diagnostics.get("rerank_top_m"),
            "category_bonus": rerank_diagnostics.get("category_bonus"),
            "predicted_query_category": rerank_diagnostics.get("predicted_query_category"),
            "candidate_doc_count": len(candidate_doc_map),
        },
        "relevant_docs": relevant_docs,
        "relevant_docs_hurt_by_reranker": demoted_relevant_rows[:snapshot_limit],
        "false_positives_promoted_by_reranker": promoted_non_relevant_rows[:snapshot_limit],
        "top_before_rerank": [
            build_ranked_doc_debug_row(
                doc_id,
                relevant_doc_id_set,
                candidate_doc_map,
                predicted_query_category,
                rank,
                reranked_positions.get(str(doc_id)),
            )
            for rank, doc_id in enumerate(query_detail["base_retrieved_doc_ids"][:snapshot_limit], start=1)
        ],
        "top_after_rerank": [
            build_ranked_doc_debug_row(
                doc_id,
                relevant_doc_id_set,
                candidate_doc_map,
                predicted_query_category,
                base_positions.get(str(doc_id)),
                rank,
            )
            for rank, doc_id in enumerate(query_detail["retrieved_doc_ids"][:snapshot_limit], start=1)
        ],
        "doc_category_summary": format_category_percentages(query_detail["retrieved_doc_ids"][:snapshot_limit], doc_category_map),
    }
