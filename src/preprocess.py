"""Text preprocessing utilities used by retrieval models."""

from __future__ import annotations

from typing import List

import pandas as pd


def _value_to_text(value: object) -> str:
    """Convert a single cell value to normalized text.

    - ``NaN`` and ``None`` become an empty string.
    - lists/tuples are joined with spaces.
    - other values are cast to strings.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)

    if pd.isna(value):
        return ""

    return str(value)


def create_content_column(df: pd.DataFrame, columns_to_merge: List[str]) -> pd.DataFrame:
    """Create normalized ``content`` and string ``id`` columns.

    Missing columns from ``columns_to_merge`` are created as empty strings.
    """
    processed = df.copy()

    for column in columns_to_merge:
        if column not in processed.columns:
            # Keep one shared preprocessing path for docs and queries.
            processed[column] = ""

    merged_values = []
    for _, row in processed[columns_to_merge].iterrows():
        parts = [_value_to_text(row[column]) for column in columns_to_merge]
        # Lowercasing here keeps lexical models (TF-IDF/BM25) consistent.
        merged_values.append(" ".join(part for part in parts if part).strip().lower())

    processed["content"] = merged_values
    if "id" in processed.columns:
        processed["id"] = processed["id"].astype(str)

    return processed
