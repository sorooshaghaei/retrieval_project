"""Pipeline orchestration for training evaluation and Kaggle submissions."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from src.evaluation import evaluate_retrieval
from src.models import run_bm25_search, run_tfidf_search
from src.preprocess import create_content_column
from src.utils import (
    build_qrels_lookup,
    load_data,
    validate_submission_against_template,
    write_kaggle_submission,
)


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
TEAM_NAME = "SeaFour"

EVAL_K = 10
SUBMISSION_K = 100
EVAL_MODELS = ("tfidf", "bm25")
SUBMISSION_MODELS = ("tfidf", "bm25")
FINAL_MODEL = "bm25"


ModelFn = Callable[..., List[dict]]


def get_model_registry() -> Dict[str, ModelFn]:
    """Return available retrieval models."""
    return {
        "tfidf": run_tfidf_search,
        "bm25": run_bm25_search,
    }


def run_pipeline() -> None:
    """Run full training evaluation and test submission generation."""
    # 1) Load raw competition files and normalize qrels format.
    docs_df, train_queries_df, test_queries_df, qrels_raw = load_data(DATA_DIR)
    qrels = build_qrels_lookup(qrels_raw)

    # 2) Build a unified text field ("content") used by all retrieval models.
    docs_df = create_content_column(docs_df, ["title", "text", "tags"])
    train_queries_df = create_content_column(train_queries_df, ["title", "text"])
    test_queries_df = create_content_column(test_queries_df, ["title", "text"])

    model_registry = get_model_registry()

    # 3) Evaluate lexical baselines on train queries with ground-truth relevance.
    print(f"\n=== Evaluation on training queries @ {EVAL_K} ===")
    for model_name in EVAL_MODELS:
        model_fn = model_registry[model_name]
        results = model_fn(docs_df, train_queries_df, top_k=EVAL_K)

        metrics = evaluate_retrieval(results, qrels, k=EVAL_K)
        print(
            f"{model_name.upper():>5} | "
            f"Precision@{EVAL_K}: {metrics['avg_precision']:.4f} | "
            f"Recall@{EVAL_K}: {metrics['avg_recall']:.4f} | "
            f"MRR@{EVAL_K}: {metrics['mrr']:.4f} | "
            f"MAP@{EVAL_K}: {metrics['map']:.4f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    template_path = DATA_DIR / "submission.csv"

    # 4) Generate Kaggle-formatted candidate submissions for test queries.
    print(f"\n=== Generating test submissions @ {SUBMISSION_K} ===")
    test_results_by_model: Dict[str, List[dict]] = {}

    for model_name in SUBMISSION_MODELS:
        model_fn = model_registry[model_name]
        test_results = model_fn(docs_df, test_queries_df, top_k=SUBMISSION_K)
        test_results_by_model[model_name] = test_results

        output_path = OUTPUT_DIR / f"solutions_{TEAM_NAME}_{model_name}.csv"
        write_kaggle_submission(test_results, template_path, output_path)
        print(f"Saved {model_name.upper()} submission: {output_path}")

    if FINAL_MODEL not in test_results_by_model and FINAL_MODEL in model_registry:
        print(f"Running {FINAL_MODEL} retrieval for final file (this may take a while)...")
        final_results = model_registry[FINAL_MODEL](
            docs_df,
            test_queries_df,
            top_k=SUBMISSION_K,
        )
        test_results_by_model[FINAL_MODEL] = final_results

    if FINAL_MODEL not in test_results_by_model:
        raise ValueError(
            f"FINAL_MODEL='{FINAL_MODEL}' is not available. "
            f"Choose one of: {sorted(model_registry)}"
        )

    final_output_path = OUTPUT_DIR / f"solutions_{TEAM_NAME}.csv"
    write_kaggle_submission(
        test_results_by_model[FINAL_MODEL],
        template_path,
        final_output_path,
    )
    print(f"Saved final upload file ({FINAL_MODEL.upper()}): {final_output_path}")

    # 5) Validate final file against template before upload.
    validation = validate_submission_against_template(final_output_path, template_path)
    print("Final submission validation:", validation)
    if not validation["is_valid"]:
        raise ValueError("Generated final submission file does not match the template requirements.")
