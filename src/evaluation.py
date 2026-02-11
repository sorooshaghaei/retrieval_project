# Metrics calculations will go here (Precision/Recall)
import numpy as np
def evaluate_retrieval(results, qrels, k):
    """
    results: list of {"query_id": str, "relevant_docs": [str,...]}
    qrels: dict { query_id (str) : [doc_id (str), ...] }
    """
    recalls = []
    precisions = []
    mrrs = []
    for res in results:
        query_id = str(res['query_id'])
        retrieved = [str(d) for d in res['relevant_docs'][:k]]
        relevant = set(qrels.get(query_id, []))
        if len(relevant) == 0:
            # If there is no ground-truth for this query, skip it.
            continue
        retrieved_set = set(retrieved)
        recall = len(retrieved_set & relevant) / len(relevant)
        precision = len(retrieved_set & relevant) / k
        recalls.append(recall)
        precisions.append(precision)
        rr = 0.0
        for rank, doc in enumerate(retrieved, 1):
            if str(doc) in relevant:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
    return {
        'avg_recall': np.mean(recalls) if recalls else float('nan'),
        'avg_precision': np.mean(precisions) if precisions else float('nan'),
        'mrr': np.mean(mrrs) if mrrs else float('nan')
    }