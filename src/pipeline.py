"""Pipeline orchestration for training evaluation and Kaggle submissions."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from src.config import DEFAULT_CONFIG, PipelineConfig
from src.evaluation import evaluate_retrieval
from src.models import (
    run_bm25_search,
    run_embedding_hybrid_search,
    run_tfidf_search,
)
from src.preprocess import create_content_column
from src.utils import (
    build_qrels_lookup,
    load_data,
    validate_submission_against_template,
    write_kaggle_submission,
)


ModelFn = Callable[..., List[dict]]


def get_model_registry() -> Dict[str, ModelFn]:
    """Return available retrieval models."""
    return {
        "tfidf": run_tfidf_search,
        "bm25": run_bm25_search,
        "embedding_hybrid": run_embedding_hybrid_search,
    }


def _prepare_datasets(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Load raw data and build the normalized text fields used by every model."""
    docs_df, train_queries_df, test_queries_df, qrels_raw = load_data(data_dir)
    qrels = build_qrels_lookup(qrels_raw)

    docs_df = create_content_column(docs_df, ["title", "text", "tags"])
    train_queries_df = create_content_column(train_queries_df, ["title", "text"])
    test_queries_df = create_content_column(test_queries_df, ["title", "text"])

    return docs_df, train_queries_df, test_queries_df, qrels


def _print_metrics(model_name: str, metrics: dict[str, float], k: int, suffix: str = "") -> None:
    """Format retrieval metrics consistently for console output."""
    label = f"{model_name.upper():>5}"
    if model_name == "embedding_hybrid":
        label = "EMBEDDING_HYBRID"

    print(
        f"{label} | "
        f"Precision@{k}: {metrics['avg_precision']:.4f} | "
        f"Recall@{k}: {metrics['avg_recall']:.4f} | "
        f"MRR@{k}: {metrics['mrr']:.4f} | "
        f"MAP@{k}: {metrics['map']:.4f}"
        f"{suffix}"
    )


def _evaluate_models(
    docs_df: pd.DataFrame,
    train_queries_df: pd.DataFrame,
    qrels: dict[str, list[str]],
    model_registry: Dict[str, ModelFn],
    config: PipelineConfig,
) -> None:
    """Evaluate configured models on the train split."""
    print(f"\n=== Evaluation on training queries @ {config.eval_k} ===")

    for model_name in config.eval_models:
        model_fn = model_registry[model_name]
        results = model_fn(docs_df, train_queries_df, top_k=config.eval_k)
        metrics = evaluate_retrieval(results, qrels, k=config.eval_k)
        _print_metrics(model_name, metrics, config.eval_k)

    if not config.run_embedding_hybrid_eval:
        print(
            "EMBEDDING_HYBRID | Skipped in main pipeline "
            "(set run_embedding_hybrid_eval=True in PipelineConfig to include hybrid comparison)."
        )
        return

    hybrid_eval_queries = train_queries_df.head(config.embedding_hybrid_eval_query_limit).copy()
    hybrid_results = model_registry["embedding_hybrid"](
        docs_df,
        hybrid_eval_queries,
        top_k=config.eval_k,
    )
    hybrid_qids = hybrid_eval_queries["id"].astype(str).tolist()
    hybrid_qrels_subset = {query_id: qrels.get(query_id, []) for query_id in hybrid_qids}
    hybrid_metrics = evaluate_retrieval(hybrid_results, hybrid_qrels_subset, k=config.eval_k)
    _print_metrics(
        "embedding_hybrid",
        hybrid_metrics,
        config.eval_k,
        suffix=f" (queries={len(hybrid_eval_queries)})",
    )


def _generate_submissions(
    docs_df: pd.DataFrame,
    test_queries_df: pd.DataFrame,
    model_registry: Dict[str, ModelFn],
    config: PipelineConfig,
) -> Path:
    """Generate per-model submissions and the final upload CSV."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    template_path = config.data_dir / "submission.csv"

    print(f"\n=== Generating test submissions @ {config.submission_k} ===")
    test_results_by_model: Dict[str, List[dict]] = {}

    for model_name in config.submission_models:
        model_fn = model_registry[model_name]
        test_results = model_fn(docs_df, test_queries_df, top_k=config.submission_k)
        test_results_by_model[model_name] = test_results

        output_path = config.output_dir / f"solutions_{config.team_name}_{model_name}.csv"
        write_kaggle_submission(test_results, template_path, output_path)
        print(f"Saved {model_name.upper()} submission: {output_path}")

    if config.final_model not in test_results_by_model and config.final_model in model_registry:
        print(f"Running {config.final_model} retrieval for final file (this may take a while)...")
        final_results = model_registry[config.final_model](
            docs_df,
            test_queries_df,
            top_k=config.submission_k,
        )
        test_results_by_model[config.final_model] = final_results

    if config.final_model not in test_results_by_model:
        raise ValueError(
            f"final_model='{config.final_model}' is not available. "
            f"Choose one of: {sorted(model_registry)}"
        )

    final_output_path = config.output_dir / f"solutions_{config.team_name}.csv"
    write_kaggle_submission(
        test_results_by_model[config.final_model],
        template_path,
        final_output_path,
    )
    print(f"Saved final upload file ({config.final_model.upper()}): {final_output_path}")

    return final_output_path


def run_pipeline(config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """Run full training evaluation and test submission generation."""
    docs_df, train_queries_df, test_queries_df, qrels = _prepare_datasets(config.data_dir)
    model_registry = get_model_registry()

    _evaluate_models(docs_df, train_queries_df, qrels, model_registry, config)
    final_output_path = _generate_submissions(docs_df, test_queries_df, model_registry, config)

    validation = validate_submission_against_template(final_output_path, config.data_dir / "submission.csv")
    print("Final submission validation:", validation)
    if not validation["is_valid"]:
        raise ValueError("Generated final submission file does not match the template requirements.")
