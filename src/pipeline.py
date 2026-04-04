from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from .categorization import build_doc_category_map, build_or_load_category_classifier, predict_category_map
from .config import AllConfig, DEFAULT_CONFIG
from .cross_encoder import build_or_load_cross_encoder
from .data import build_content_frame, build_query_classifier_frame, ensure_unique_ids, load_json_frame, require_columns
from .evaluation import compute_category_accuracy
from .infra.runtime import resolve_runtime_paths
from .output.submission import write_kaggle_submission
from .reranking import rerank_results_with_cross_encoder
from .retrieval import prepare_retriever, run_category_filtered_retrieval, run_retrieval
from .types import GroundTruthEntry, ModelName, RetrievalResult, RuntimePaths


@dataclass
class LoadedFrames:
    docs_raw: pd.DataFrame
    train_queries_raw: pd.DataFrame
    test_queries_raw: pd.DataFrame
    sample_submission: pd.DataFrame
    docs: pd.DataFrame
    train_queries: pd.DataFrame
    test_queries: pd.DataFrame
    docs_classifier: pd.DataFrame
    train_queries_classifier: pd.DataFrame
    test_queries_classifier: pd.DataFrame


@dataclass
class CategoryArtifacts:
    classifier_artifacts: dict[str, Any] | None
    train_query_category_map: dict[str, str] | None
    test_query_category_map: dict[str, str] | None
    doc_category_map: dict[str, Any] | None
    classifier_accuracy: float


def bootstrap(output_filename: str = "solutions_SeaFour.csv", config: AllConfig = DEFAULT_CONFIG) -> tuple[RuntimePaths, AllConfig]:
    paths = resolve_runtime_paths(output_filename=output_filename)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    return paths, config


def load_project_frames(paths: RuntimePaths, config: AllConfig = DEFAULT_CONFIG) -> LoadedFrames:
    docs_raw_df = load_json_frame(paths.data_dir / "docs.json", "Documents")
    train_queries_raw_df = load_json_frame(paths.data_dir / "queries_train.json", "Train queries")
    test_queries_raw_df = load_json_frame(paths.data_dir / "queries_test.json", "Test queries")
    sample_submission_df = pd.read_csv(paths.data_dir / "submission.csv")

    require_columns(docs_raw_df, ["id", "title", "text", "tags", "category"], "Documents")
    require_columns(train_queries_raw_df, ["id", "title", "text", "tags", "category"], "Train queries")
    require_columns(test_queries_raw_df, ["id", "title", "text", "tags"], "Test queries")
    require_columns(sample_submission_df, ["query_id", "relevant_doc_ids", "category"], "Sample submission")
    ensure_unique_ids(docs_raw_df, "Documents")
    ensure_unique_ids(train_queries_raw_df, "Train queries")
    ensure_unique_ids(test_queries_raw_df, "Test queries")

    docs_df = build_content_frame(docs_raw_df, config.data_columns.document_text_columns)
    train_queries_df = build_content_frame(train_queries_raw_df, config.data_columns.retrieval_query_columns)
    test_queries_df = build_content_frame(test_queries_raw_df, config.data_columns.retrieval_query_columns)
    docs_classifier_df = docs_df[["id", "content", "category"]].copy()
    train_queries_classifier_df = build_query_classifier_frame(
        train_queries_raw_df,
        include_tags=config.data_columns.use_query_tags_in_classifier,
    )
    test_queries_classifier_df = build_query_classifier_frame(
        test_queries_raw_df,
        include_tags=config.data_columns.use_query_tags_in_classifier,
    )

    if docs_df["content"].eq("").all():
        raise ValueError("All document content is empty after preprocessing. Check the source columns or normalization.")

    return LoadedFrames(
        docs_raw=docs_raw_df,
        train_queries_raw=train_queries_raw_df,
        test_queries_raw=test_queries_raw_df,
        sample_submission=sample_submission_df,
        docs=docs_df,
        train_queries=train_queries_df,
        test_queries=test_queries_df,
        docs_classifier=docs_classifier_df,
        train_queries_classifier=train_queries_classifier_df,
        test_queries_classifier=test_queries_classifier_df,
    )


def prepare_retrievers(
    frames: LoadedFrames,
    paths: RuntimePaths,
    config: AllConfig = DEFAULT_CONFIG,
) -> dict[str, dict[str, Any]]:
    model_names = set(config.retrieval_pipeline.evaluation_models) | {config.retrieval_pipeline.final_model}
    prepared_retrievers: dict[str, dict[str, Any]] = {}
    for model_name in model_names:
        prepared_retrievers[model_name] = prepare_retriever(
            model_name=model_name,
            docs_frame=frames.docs,
            cache_dir=paths.cache_dir,
            config=config,
        )
    return prepared_retrievers


def predict_categories(
    frames: LoadedFrames,
    paths: RuntimePaths,
    ground_truth: dict[str, GroundTruthEntry] | None = None,
    config: AllConfig = DEFAULT_CONFIG,
) -> CategoryArtifacts:
    if not config.retrieval_pipeline.enable_category_prediction:
        return CategoryArtifacts(
            classifier_artifacts=None,
            train_query_category_map=None,
            test_query_category_map=None,
            doc_category_map=None,
            classifier_accuracy=0.0,
        )

    classifier_artifacts = build_or_load_category_classifier(
        train_frame=frames.docs_classifier,
        cache_dir=paths.cache_dir,
        config=config,
    )
    train_query_category_map = predict_category_map(frames.train_queries_classifier, classifier_artifacts)
    test_query_category_map = predict_category_map(frames.test_queries_classifier, classifier_artifacts)
    doc_category_map = build_doc_category_map(frames.docs)
    classifier_accuracy = (
        compute_category_accuracy(ground_truth, train_query_category_map)
        if ground_truth is not None
        else 0.0
    )
    return CategoryArtifacts(
        classifier_artifacts=classifier_artifacts,
        train_query_category_map=train_query_category_map,
        test_query_category_map=test_query_category_map,
        doc_category_map=doc_category_map,
        classifier_accuracy=classifier_accuracy,
    )


def build_cross_encoder_reranker(
    frames: LoadedFrames,
    paths: RuntimePaths,
    ground_truth: dict[str, GroundTruthEntry],
    config: AllConfig = DEFAULT_CONFIG,
) -> Any | None:
    if not config.retrieval_pipeline.enable_cross_encoder_rerank:
        return None
    return build_or_load_cross_encoder(
        train_queries_frame=frames.train_queries,
        docs_frame=frames.docs,
        ground_truth=ground_truth,
        cache_dir=paths.cache_dir,
        config=config,
    )


def run_first_stage_retrieval(
    frames: LoadedFrames,
    paths: RuntimePaths,
    prepared_retrievers: dict[str, dict[str, Any]],
    category_artifacts: CategoryArtifacts,
    split: Literal["train", "test"] = "test",
    top_k: int | None = None,
    model_name_override: ModelName | None = None,
    config: AllConfig = DEFAULT_CONFIG,
) -> tuple[list[RetrievalResult], dict[str, str] | None]:
    query_frame = frames.train_queries if split == "train" else frames.test_queries
    embedding_kind = "queries_train" if split == "train" else "queries_test"
    retrieval_top_k = config.retrieval_pipeline.submit_top_k if top_k is None else top_k
    selected_model = model_name_override or config.retrieval_pipeline.final_model
    category_predictions = (
        category_artifacts.train_query_category_map
        if split == "train"
        else category_artifacts.test_query_category_map
    )

    if config.retrieval_pipeline.enable_category_filter and category_artifacts.classifier_artifacts is not None:
        results, _ = run_category_filtered_retrieval(
            docs_frame=frames.docs,
            queries_frame=query_frame,
            classifier_artifacts=category_artifacts.classifier_artifacts,
            top_k=retrieval_top_k,
            cache_dir=paths.cache_dir,
            model_name=selected_model,
            prepared_artifacts=prepared_retrievers.get(selected_model),
            query_category_map=category_predictions,
            embedding_kind_prefix=embedding_kind,
            config=config,
        )
        return results, category_predictions

    results = run_retrieval(
        model_name=selected_model,
        docs_frame=frames.docs,
        queries_frame=query_frame,
        top_k=retrieval_top_k,
        cache_dir=paths.cache_dir,
        prepared_artifacts=prepared_retrievers[selected_model],
        embedding_kind=embedding_kind,
        config=config,
    )
    return results, category_predictions


def rerank_retrieval_results(
    results: list[RetrievalResult],
    frames: LoadedFrames,
    category_artifacts: CategoryArtifacts,
    cross_encoder_reranker: Any | None,
    split: Literal["train", "test"] = "test",
    config: AllConfig = DEFAULT_CONFIG,
) -> list[RetrievalResult]:
    if cross_encoder_reranker is None:
        return results
    query_category_map = (
        category_artifacts.train_query_category_map
        if split == "train"
        else category_artifacts.test_query_category_map
    )
    if query_category_map is None or category_artifacts.doc_category_map is None:
        return results
    query_frame = frames.train_queries if split == "train" else frames.test_queries
    return rerank_results_with_cross_encoder(
        results=results,
        query_frame=query_frame,
        docs_frame=frames.docs,
        cross_encoder=cross_encoder_reranker,
        query_category_map=query_category_map,
        doc_category_map=category_artifacts.doc_category_map,
        infer_batch_size=config.cross_encoder.infer_batch_size,
        rerank_top_m=config.cross_encoder.rerank_top_m,
        category_bonus=config.cross_encoder.category_bonus if config.retrieval_pipeline.enable_category_filter else 0.0,
    )


def write_submission(
    results: list[RetrievalResult],
    paths: RuntimePaths,
    category_predictions: dict[str, str] | None = None,
) -> None:
    write_kaggle_submission(
        results=results,
        sample_csv_path=paths.data_dir / "submission.csv",
        output_csv_path=paths.output_path,
        category_predictions=category_predictions,
    )
