from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import GroundTruthEntry, RetrievalResult


def load_ground_truth(path: Path) -> dict[str, GroundTruthEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw_ground_truth = json.load(handle)
    ground_truth: dict[str, GroundTruthEntry] = {}
    for query_id, info in raw_ground_truth.items():
        relevant_items = info.get("relevant_doc_ids", [])
        ground_truth[str(query_id)] = {
            "relevant_doc_ids": {str(item["doc_id"]) for item in relevant_items},
            "total_relevant_docs": int(info.get("total_relevant_docs", len(relevant_items))),
            "category": info.get("category"),
        }
    return ground_truth


def recall_at_k(results: list[RetrievalResult], ground_truth: dict[str, GroundTruthEntry], k: int) -> float:
    recalls: list[float] = []
    for item in results:
        query_id = str(item["query_id"])
        if query_id not in ground_truth:
            continue
        relevant_doc_ids = ground_truth[query_id]["relevant_doc_ids"]
        total_relevant_docs = ground_truth[query_id]["total_relevant_docs"]
        predicted_doc_ids = item["relevant_docs"][:k]
        hits = sum(doc_id in relevant_doc_ids for doc_id in predicted_doc_ids)
        recalls.append(hits / total_relevant_docs if total_relevant_docs > 0 else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def precision_at_k(results: list[RetrievalResult], ground_truth: dict[str, GroundTruthEntry], k: int) -> float:
    precisions: list[float] = []
    for item in results:
        query_id = str(item["query_id"])
        if query_id not in ground_truth:
            continue
        relevant_doc_ids = ground_truth[query_id]["relevant_doc_ids"]
        predicted_doc_ids = item["relevant_docs"][:k]
        if not predicted_doc_ids:
            precisions.append(0.0)
            continue
        hits = sum(doc_id in relevant_doc_ids for doc_id in predicted_doc_ids)
        precisions.append(hits / len(predicted_doc_ids))
    return float(np.mean(precisions)) if precisions else 0.0


def mrr_at_k(results: list[RetrievalResult], ground_truth: dict[str, GroundTruthEntry], k: int) -> float:
    reciprocal_ranks: list[float] = []
    for item in results:
        query_id = str(item["query_id"])
        if query_id not in ground_truth:
            continue
        relevant_doc_ids = ground_truth[query_id]["relevant_doc_ids"]
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(item["relevant_docs"][:k], start=1):
            if doc_id in relevant_doc_ids:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def compute_category_accuracy(
    ground_truth: dict[str, GroundTruthEntry],
    predicted_categories: dict[str, str] | None,
    default_if_missing: float = 0.0,
) -> float:
    if predicted_categories is None:
        return float(default_if_missing)
    try:
        from sklearn.metrics import accuracy_score
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `scikit-learn`. Install it with `%pip install scikit-learn`."
        ) from exc
    y_true: list[str] = []
    y_pred: list[str] = []
    for query_id, info in ground_truth.items():
        true_category = info.get("category")
        predicted_category = predicted_categories.get(str(query_id))
        if true_category is None or predicted_category is None:
            continue
        y_true.append(str(true_category))
        y_pred.append(str(predicted_category))
    if not y_true:
        return float(default_if_missing)
    return float(accuracy_score(y_true, y_pred))


def leaderboard_score(
    results: list[RetrievalResult],
    ground_truth: dict[str, GroundTruthEntry],
    k: int,
    predicted_categories: dict[str, str] | None = None,
    accuracy_value: float | None = None,
) -> dict[str, float]:
    recall_value = recall_at_k(results, ground_truth, k=k)
    precision_value = precision_at_k(results, ground_truth, k=k)
    mrr_value = mrr_at_k(results, ground_truth, k=k)
    category_accuracy = (
        float(accuracy_value)
        if accuracy_value is not None
        else compute_category_accuracy(ground_truth, predicted_categories)
    )
    combined_score = 0.25 * (recall_value + precision_value + mrr_value + category_accuracy)
    return {
        "Recall": recall_value,
        "Precision": precision_value,
        "MRR": mrr_value,
        "Accuracy": category_accuracy,
        "LeaderboardScore": combined_score,
    }
