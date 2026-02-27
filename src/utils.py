"""Utility helpers for data loading and Kaggle submission formatting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import pandas as pd


def load_data(data_dir: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load docs, train queries, test queries, and raw qrels JSON.

    Expected files inside ``data_dir``:
    - docs.json
    - queries_train.json
    - queries_test.json
    - qgts_train.json
    """
    data_dir = Path(data_dir)

    docs_path = data_dir / "docs.json"
    train_queries_path = data_dir / "queries_train.json"
    test_queries_path = data_dir / "queries_test.json"
    qrels_path = data_dir / "qgts_train.json"

    # JSON files are loaded as DataFrames for easier model preprocessing.
    docs_df = pd.read_json(docs_path)
    train_queries_df = pd.read_json(train_queries_path)
    test_queries_df = pd.read_json(test_queries_path)

    with qrels_path.open("r", encoding="utf-8") as file:
        qrels_raw = json.load(file)

    return docs_df, train_queries_df, test_queries_df, qrels_raw


def build_qrels_lookup(qrels_raw: Mapping) -> Dict[str, List[str]]:
    """Convert raw qrels format into ``query_id -> [relevant_doc_id, ...]``."""
    qrels_lookup: Dict[str, List[str]] = {}

    for query_id, query_info in qrels_raw.items():
        relevant_items = query_info.get("relevant_doc_ids", []) if isinstance(query_info, dict) else []
        # Keep only doc ids; relevance scores are not currently weighted in metrics.
        qrels_lookup[str(query_id)] = [str(item["doc_id"]) for item in relevant_items if "doc_id" in item]

    return qrels_lookup


def write_kaggle_submission(
    retrieval_results: Sequence[Mapping[str, object]],
    sample_csv_path: Path | str,
    output_csv_path: Path | str,
) -> None:
    """Write Kaggle submission CSV using the same row order as sample CSV.

    Kaggle expects column 2 (``relevant_doc_ids``) to be a JSON-encoded list.
    """
    sample_csv_path = Path(sample_csv_path)
    output_csv_path = Path(output_csv_path)

    predictions_by_query: Dict[str, List[str]] = {}
    for item in retrieval_results:
        query_id = str(item["query_id"])
        predicted_docs = [str(doc_id) for doc_id in item["relevant_docs"]]
        predictions_by_query[query_id] = predicted_docs

    with sample_csv_path.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None or len(fieldnames) < 2:
        raise ValueError("Sample submission must have at least 2 columns.")

    query_id_col = fieldnames[0]
    prediction_col = fieldnames[1]
    category_col = fieldnames[2] if len(fieldnames) >= 3 else None

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            query_id = str(row[query_id_col])
            if query_id not in predictions_by_query:
                raise ValueError(f"Missing prediction for query_id: {query_id}")

            # Kaggle expects a JSON string in the relevant_doc_ids column.
            output_row: MutableMapping[str, str] = {
                query_id_col: query_id,
                prediction_col: json.dumps(predictions_by_query[query_id]),
            }

            if category_col is not None:
                output_row[category_col] = row.get(category_col, "?") or "?"

            writer.writerow(output_row)


def validate_submission_against_template(
    submission_csv_path: Path | str,
    template_csv_path: Path | str,
) -> Dict[str, object]:
    """Validate submission structure and query ordering against template CSV."""
    submission_csv_path = Path(submission_csv_path)
    template_csv_path = Path(template_csv_path)

    template_df = pd.read_csv(template_csv_path)
    submission_df = pd.read_csv(submission_csv_path)

    result: Dict[str, object] = {
        "submission_file": str(submission_csv_path.name),
        "column_match": list(submission_df.columns) == list(template_df.columns),
        "row_count_match": len(submission_df) == len(template_df),
        "query_id_order_match": (
            submission_df.iloc[:, 0].astype(str).tolist()
            == template_df.iloc[:, 0].astype(str).tolist()
        ),
        "valid_json_lists": True,
        "min_k": None,
        "max_k": None,
    }

    doc_list_lengths: List[int] = []
    try:
        for value in submission_df.iloc[:, 1].astype(str):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                result["valid_json_lists"] = False
                break
            doc_list_lengths.append(len(parsed))
    except Exception:
        result["valid_json_lists"] = False

    if doc_list_lengths:
        result["min_k"] = min(doc_list_lengths)
        result["max_k"] = max(doc_list_lengths)

    result["is_valid"] = bool(
        result["column_match"]
        and result["row_count_match"]
        and result["query_id_order_match"]
        and result["valid_json_lists"]
    )

    return result
