from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import AllConfig, DEFAULT_CONFIG
from ..models.embeddings import _load_or_encode_embeddings, _load_sentence_model
from ..types import ModelName, RetrievalResult
from .indexes import build_or_load_bm25_index, build_or_load_tfidf_index


@dataclass(frozen=True)
class ScoredRetrievalOutput:
    results: list[RetrievalResult]
    topk_indices: np.ndarray
    topk_scores: np.ndarray
    elapsed_seconds: float


def validate_pipeline_settings(document_count: int, config: AllConfig = DEFAULT_CONFIG) -> None:
    final_model = config.retrieval_pipeline.final_model
    evaluation_models = config.retrieval_pipeline.evaluation_models
    if final_model not in {"tfidf", "bm25", "embedding"}:
        raise ValueError(f"Unknown final_model: {final_model}")
    if any(model_name not in {"tfidf", "bm25", "embedding"} for model_name in evaluation_models):
        raise ValueError(f"Unknown model in evaluation_models: {evaluation_models}")
    if document_count <= 0:
        raise ValueError("The document collection is empty.")
    if config.retrieval_pipeline.submit_top_k <= 0:
        raise ValueError("submit_top_k must be positive.")


def top_k_indices(score_vector: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    capped_top_k = min(top_k, score_vector.shape[0])
    if capped_top_k == score_vector.shape[0]:
        return np.argsort(score_vector)[::-1]
    candidate_indices = np.argpartition(score_vector, -capped_top_k)[-capped_top_k:]
    return candidate_indices[np.argsort(score_vector[candidate_indices])[::-1]]


def truncate_results(results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    return [{"query_id": item["query_id"], "relevant_docs": item["relevant_docs"][:top_k]} for item in results]


def progress_interval(total_items: int, target_updates: int = 5) -> int:
    return max(1, total_items // max(1, target_updates))


def _top_k_indices_and_scores(score_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    top_indices = top_k_indices(score_vector, top_k)
    top_scores = np.asarray(score_vector[top_indices], dtype=np.float32)
    return np.asarray(top_indices, dtype=np.int64), top_scores


def _stack_topk_rows(topk_rows: list[np.ndarray], dtype: Any) -> np.ndarray:
    if not topk_rows:
        return np.empty((0, 0), dtype=dtype)
    return np.vstack(topk_rows).astype(dtype, copy=False)


def prepare_retriever(
    model_name: ModelName,
    docs_frame: pd.DataFrame,
    cache_dir: Path,
    config: AllConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if model_name == "tfidf":
        return build_or_load_tfidf_index(docs_frame, cache_dir=cache_dir, config=config)
    if model_name == "bm25":
        return build_or_load_bm25_index(docs_frame, cache_dir=cache_dir, config=config)
    if model_name == "embedding":
        model = _load_sentence_model(config.embedding.model_name, cache_dir=cache_dir, config=config)
        doc_embeddings = _load_or_encode_embeddings(
            docs_frame,
            kind="docs",
            model=model,
            model_name=config.embedding.model_name,
            batch_size=config.embedding.batch_size,
            cache_dir=cache_dir,
            config=config,
        )
        return {"model": model, "doc_embeddings": doc_embeddings, "doc_ids": docs_frame["id"].to_numpy()}
    raise ValueError(f"Unknown model: {model_name}")


def run_tfidf_search(
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    top_k: int,
    cache_dir: Path,
    prepared_artifacts: dict[str, Any] | None = None,
    config: AllConfig = DEFAULT_CONFIG,
) -> list[RetrievalResult]:
    return run_tfidf_search_scored(
        docs_frame=docs_frame,
        queries_frame=queries_frame,
        top_k=top_k,
        cache_dir=cache_dir,
        prepared_artifacts=prepared_artifacts,
        config=config,
    ).results


def run_tfidf_search_scored(
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    top_k: int,
    cache_dir: Path,
    prepared_artifacts: dict[str, Any] | None = None,
    config: AllConfig = DEFAULT_CONFIG,
) -> ScoredRetrievalOutput:
    artifacts = prepared_artifacts or build_or_load_tfidf_index(docs_frame, cache_dir=cache_dir, config=config)
    vectorizer = artifacts["vectorizer"]
    doc_vectors = artifacts["doc_vectors"]
    doc_ids = artifacts["doc_ids"]
    query_vectors = vectorizer.transform(queries_frame["content"])
    query_ids = queries_frame["id"].astype(str).tolist()
    capped_top_k = min(top_k, len(doc_ids))
    log_every = progress_interval(len(query_ids))
    print(f"  [TF-IDF] vectorized {len(query_ids):,} queries against {len(doc_ids):,} docs with capped_top_k={capped_top_k:,}")
    start_time = time.perf_counter()
    results: list[RetrievalResult] = []
    topk_indices_rows: list[np.ndarray] = []
    topk_score_rows: list[np.ndarray] = []
    for row_index, query_id in enumerate(query_ids):
        score_row = query_vectors[row_index] @ doc_vectors.T
        score_vector = np.asarray(score_row.toarray()).ravel()
        top_indices, top_scores = _top_k_indices_and_scores(score_vector, capped_top_k)
        results.append({"query_id": query_id, "relevant_docs": doc_ids[top_indices].tolist()})
        topk_indices_rows.append(top_indices)
        topk_score_rows.append(top_scores)
        if (row_index + 1) % log_every == 0 or row_index == len(query_ids) - 1:
            print(f"  [TF-IDF] processed {row_index + 1:,}/{len(query_ids):,} queries")
    elapsed_seconds = time.perf_counter() - start_time
    return ScoredRetrievalOutput(
        results=results,
        topk_indices=_stack_topk_rows(topk_indices_rows, np.int64),
        topk_scores=_stack_topk_rows(topk_score_rows, np.float32),
        elapsed_seconds=float(elapsed_seconds),
    )


def run_bm25_search(
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    top_k: int,
    cache_dir: Path,
    prepared_artifacts: dict[str, Any] | None = None,
    config: AllConfig = DEFAULT_CONFIG,
) -> list[RetrievalResult]:
    return run_bm25_search_scored(
        docs_frame=docs_frame,
        queries_frame=queries_frame,
        top_k=top_k,
        cache_dir=cache_dir,
        prepared_artifacts=prepared_artifacts,
        config=config,
    ).results


def run_bm25_search_scored(
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    top_k: int,
    cache_dir: Path,
    prepared_artifacts: dict[str, Any] | None = None,
    config: AllConfig = DEFAULT_CONFIG,
) -> ScoredRetrievalOutput:
    artifacts = prepared_artifacts or build_or_load_bm25_index(docs_frame, cache_dir=cache_dir, config=config)
    bm25 = artifacts["bm25"]
    doc_ids = artifacts["doc_ids"]
    capped_top_k = min(top_k, len(doc_ids))
    query_pairs = list(queries_frame[["id", "content"]].itertuples(index=False, name=None))
    log_every = progress_interval(len(query_pairs))
    print(f"  [BM25+] scoring {len(query_pairs):,} queries against {len(doc_ids):,} docs with capped_top_k={capped_top_k:,}")
    start_time = time.perf_counter()
    results: list[RetrievalResult] = []
    topk_indices_rows: list[np.ndarray] = []
    topk_score_rows: list[np.ndarray] = []
    for row_index, (query_id, query_text) in enumerate(query_pairs):
        score_vector = np.asarray(bm25.get_scores(query_text.split()), dtype=np.float32)
        top_indices, top_scores = _top_k_indices_and_scores(score_vector, capped_top_k)
        results.append({"query_id": str(query_id), "relevant_docs": doc_ids[top_indices].tolist()})
        topk_indices_rows.append(top_indices)
        topk_score_rows.append(top_scores)
        if (row_index + 1) % log_every == 0 or row_index == len(query_pairs) - 1:
            print(f"  [BM25+] processed {row_index + 1:,}/{len(query_pairs):,} queries")
    elapsed_seconds = time.perf_counter() - start_time
    return ScoredRetrievalOutput(
        results=results,
        topk_indices=_stack_topk_rows(topk_indices_rows, np.int64),
        topk_scores=_stack_topk_rows(topk_score_rows, np.float32),
        elapsed_seconds=float(elapsed_seconds),
    )


def run_embedding_search(
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    top_k: int,
    cache_dir: Path,
    prepared_artifacts: dict[str, Any] | None = None,
    embedding_kind: str = "queries",
    config: AllConfig = DEFAULT_CONFIG,
) -> list[RetrievalResult]:
    return run_embedding_search_scored(
        docs_frame=docs_frame,
        queries_frame=queries_frame,
        top_k=top_k,
        cache_dir=cache_dir,
        prepared_artifacts=prepared_artifacts,
        embedding_kind=embedding_kind,
        config=config,
    ).results


def run_embedding_search_scored(
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    top_k: int,
    cache_dir: Path,
    prepared_artifacts: dict[str, Any] | None = None,
    embedding_kind: str = "queries",
    config: AllConfig = DEFAULT_CONFIG,
) -> ScoredRetrievalOutput:
    artifacts = prepared_artifacts or prepare_retriever("embedding", docs_frame, cache_dir=cache_dir, config=config)
    model = artifacts["model"]
    doc_embeddings = artifacts["doc_embeddings"]
    doc_ids = artifacts["doc_ids"]
    query_embeddings = _load_or_encode_embeddings(
        queries_frame,
        kind=embedding_kind,
        model=model,
        model_name=config.embedding.model_name,
        batch_size=config.embedding.batch_size,
        cache_dir=cache_dir,
        config=config,
    )
    query_ids = queries_frame["id"].astype(str).tolist()
    capped_top_k = min(top_k, len(doc_ids))
    results: list[RetrievalResult] = []
    topk_indices_rows: list[np.ndarray] = []
    topk_score_rows: list[np.ndarray] = []
    chunk_size = config.embedding.query_chunk_size
    total_chunks = (len(query_embeddings) + chunk_size - 1) // chunk_size
    print(
        f"  [Embedding] scoring {len(query_ids):,} queries against {len(doc_ids):,} docs "
        f"with capped_top_k={capped_top_k:,}, chunk_size={chunk_size:,}, embedding_cache_key='{embedding_kind}'"
    )
    start_time = time.perf_counter()
    for chunk_index, start_index in enumerate(range(0, len(query_embeddings), chunk_size), start=1):
        stop_index = start_index + chunk_size
        print(f"  [Embedding] chunk {chunk_index:,}/{total_chunks:,}: queries {start_index + 1:,}-{min(stop_index, len(query_embeddings)):,}")
        score_block = query_embeddings[start_index:stop_index] @ doc_embeddings.T
        for row_offset, score_vector in enumerate(score_block):
            top_indices, top_scores = _top_k_indices_and_scores(score_vector, capped_top_k)
            query_id = query_ids[start_index + row_offset]
            results.append({"query_id": query_id, "relevant_docs": doc_ids[top_indices].tolist()})
            topk_indices_rows.append(top_indices)
            topk_score_rows.append(top_scores)
    elapsed_seconds = time.perf_counter() - start_time
    return ScoredRetrievalOutput(
        results=results,
        topk_indices=_stack_topk_rows(topk_indices_rows, np.int64),
        topk_scores=_stack_topk_rows(topk_score_rows, np.float32),
        elapsed_seconds=float(elapsed_seconds),
    )


def run_retrieval_scored(
    model_name: ModelName,
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    top_k: int,
    cache_dir: Path,
    prepared_artifacts: dict[str, Any] | None = None,
    embedding_kind: str = "queries",
    config: AllConfig = DEFAULT_CONFIG,
) -> ScoredRetrievalOutput:
    models: dict[ModelName, Any] = {
        "tfidf": run_tfidf_search_scored,
        "bm25": run_bm25_search_scored,
        "embedding": run_embedding_search_scored,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(models)}")
    print("=" * 88)
    print(f"Starting retrieval: model={model_name}")
    print(
        f"  parameters: top_k={top_k:,}, docs={len(docs_frame):,}, queries={len(queries_frame):,}, "
        f"prepared_artifacts={'yes' if prepared_artifacts is not None else 'no'}, embedding_kind='{embedding_kind}'"
    )
    if model_name == "embedding":
        output = models[model_name](
            docs_frame=docs_frame,
            queries_frame=queries_frame,
            top_k=top_k,
            cache_dir=cache_dir,
            prepared_artifacts=prepared_artifacts,
            embedding_kind=embedding_kind,
            config=config,
        )
    else:
        output = models[model_name](
            docs_frame=docs_frame,
            queries_frame=queries_frame,
            top_k=top_k,
            cache_dir=cache_dir,
            prepared_artifacts=prepared_artifacts,
            config=config,
        )
    print(
        f"Completed retrieval: model={model_name}, results={len(output.results):,} queries, "
        f"elapsed={output.elapsed_seconds:.1f}s"
    )
    print("=" * 88)
    return output


def run_retrieval(
    model_name: ModelName,
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    top_k: int,
    cache_dir: Path,
    prepared_artifacts: dict[str, Any] | None = None,
    embedding_kind: str = "queries",
    config: AllConfig = DEFAULT_CONFIG,
) -> list[RetrievalResult]:
    return run_retrieval_scored(
        model_name=model_name,
        docs_frame=docs_frame,
        queries_frame=queries_frame,
        top_k=top_k,
        cache_dir=cache_dir,
        prepared_artifacts=prepared_artifacts,
        embedding_kind=embedding_kind,
        config=config,
    ).results


def run_category_filtered_retrieval(
    docs_frame: pd.DataFrame,
    queries_frame: pd.DataFrame,
    classifier_artifacts: dict[str, Any],
    top_k: int,
    cache_dir: Path,
    model_name: ModelName | None = None,
    prepared_artifacts: dict[str, Any] | None = None,
    query_category_map: dict[str, str] | None = None,
    embedding_kind_prefix: str = "queries",
    config: AllConfig = DEFAULT_CONFIG,
) -> tuple[list[RetrievalResult], dict[str, str]]:
    from ..categorization.classifier import predict_category_map

    selected_model = model_name or config.retrieval_pipeline.final_model
    query_category_map = query_category_map or predict_category_map(queries_frame, classifier_artifacts)
    print(
        f"Starting category-filtered retrieval: model={selected_model}, top_k={top_k:,}, "
        f"queries={len(queries_frame):,}, predicted_categories={len(set(query_category_map.values())):,}"
    )

    category_to_positions: dict[str, list[int]] = {}
    for iloc_idx, (_, row) in enumerate(docs_frame.iterrows()):
        category_to_positions.setdefault(row["category"], []).append(iloc_idx)

    results: list[RetrievalResult] = []
    full_artifacts = prepared_artifacts
    category_artifacts_cache: dict[str, dict[str, Any]] = {}
    for category in sorted(set(query_category_map.values())):
        cat_positions_list = category_to_positions.get(category, [])
        cat_queries_frame = queries_frame[queries_frame["id"].map(query_category_map) == category]
        if not cat_positions_list:
            print(
                f"WARNING: No documents found for predicted category '{category}', "
                "falling back to full corpus for those queries."
            )
            cat_docs_frame = docs_frame
            cat_prepared_artifacts = full_artifacts or prepare_retriever(
                selected_model,
                cat_docs_frame,
                cache_dir=cache_dir,
                config=config,
            )
        else:
            cat_docs_frame = docs_frame.iloc[cat_positions_list].reset_index(drop=True)
            if category not in category_artifacts_cache:
                category_artifacts_cache[category] = prepare_retriever(
                    selected_model,
                    cat_docs_frame,
                    cache_dir=cache_dir,
                    config=config,
                )
            cat_prepared_artifacts = category_artifacts_cache[category]

        embedding_kind = f"{embedding_kind_prefix}_{category}_filtered" if selected_model == "embedding" else embedding_kind_prefix
        cat_results = run_retrieval(
            model_name=selected_model,
            docs_frame=cat_docs_frame,
            queries_frame=cat_queries_frame,
            top_k=top_k,
            cache_dir=cache_dir,
            prepared_artifacts=cat_prepared_artifacts,
            embedding_kind=embedding_kind,
            config=config,
        )
        results.extend(cat_results)

    query_order = list(queries_frame["id"].astype(str))
    results_by_query = {item["query_id"]: item for item in results}
    ordered_results = [results_by_query[query_id] for query_id in query_order if query_id in results_by_query]
    return ordered_results, query_category_map
