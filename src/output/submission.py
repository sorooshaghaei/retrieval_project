from __future__ import annotations

import csv
import json
from pathlib import Path

from ..types import RetrievalResult


def write_kaggle_submission(
    results: list[RetrievalResult],
    sample_csv_path: Path,
    output_csv_path: Path,
    category_predictions: dict[str, str] | None = None,
) -> None:
    prediction_map = {
        str(item["query_id"]): [str(doc_id) for doc_id in item["relevant_docs"]]
        for item in results
    }
    with open(sample_csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or len(fieldnames) < 2:
        raise ValueError("Invalid sample submission format.")
    query_id_column = fieldnames[0]
    prediction_column = fieldnames[1]
    category_column = fieldnames[2] if len(fieldnames) >= 3 else None
    with open(output_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            query_id = str(row[query_id_column])
            if query_id not in prediction_map:
                raise ValueError(f"Missing retrieval prediction for query_id={query_id}")
            output_row = {
                query_id_column: query_id,
                prediction_column: json.dumps(prediction_map[query_id]),
            }
            if category_column is not None:
                if category_predictions is None:
                    output_row[category_column] = row.get(category_column, "?") or "?"
                else:
                    if query_id not in category_predictions:
                        raise ValueError(f"Missing category prediction for query_id={query_id}")
                    output_row[category_column] = str(category_predictions[query_id])
            writer.writerow(output_row)
