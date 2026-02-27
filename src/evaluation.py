"""Evaluation metrics for ranked retrieval outputs."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np


def _average_precision_at_k(retrieved_docs: List[str], relevant_docs: set[str], k: int) -> float:
    """Compute AP@k for one query."""
    if not relevant_docs:
        return 0.0

    num_hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved_docs[:k], start=1):
        if doc_id in relevant_docs:
            num_hits += 1
            precision_sum += num_hits / rank

    return precision_sum / len(relevant_docs)


def evaluate_retrieval(results: Sequence[Mapping[str, object]], qrels: Mapping[str, List[str]], k: int) -> Dict[str, float]:
    """Compute retrieval metrics over all queries with available qrels.

    Returns:
        ``avg_recall``, ``avg_precision``, ``mrr``, ``map``
    """
    recalls: List[float] = []
    precisions: List[float] = []
    reciprocal_ranks: List[float] = []
    average_precisions: List[float] = []

    for item in results:
        query_id = str(item["query_id"])
        retrieved_docs = [str(doc) for doc in item["relevant_docs"][:k]]
        relevant_docs = set(str(doc) for doc in qrels.get(query_id, []))

        if not relevant_docs:
            # Skip queries that have no ground-truth labels in qrels.
            continue

        relevant_hits = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        recalls.append(relevant_hits / len(relevant_docs))
        precisions.append(relevant_hits / max(1, len(retrieved_docs)))

        # Reciprocal rank looks only at the first relevant document position.
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        average_precisions.append(_average_precision_at_k(retrieved_docs, relevant_docs, k))

    return {
        "avg_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "avg_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
        "map": float(np.mean(average_precisions)) if average_precisions else float("nan"),
    }
