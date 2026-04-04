from __future__ import annotations

import re
from typing import Any, Sequence

import pandas as pd

from ..config import DEFAULT_CONFIG
from .loading import require_columns

_TOKEN_RE = re.compile(DEFAULT_CONFIG.retrieval_pipeline.token_pattern)
_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(f"[{re.escape(DEFAULT_CONFIG.normalization.separator_chars)}]")


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    if pd.isna(value):
        return ""
    return str(value)


def normalize_text(text: Any) -> str:
    if text is None:
        cleaned_text = ""
    elif not isinstance(text, str) and pd.isna(text):
        cleaned_text = ""
    else:
        cleaned_text = str(text)

    normalization = DEFAULT_CONFIG.normalization
    if normalization.replace_separators:
        cleaned_text = _SEPARATOR_RE.sub(" ", cleaned_text)
    if normalization.lowercase:
        cleaned_text = cleaned_text.lower()
    if normalization.collapse_whitespace:
        cleaned_text = _WHITESPACE_RE.sub(" ", cleaned_text)
    if normalization.strip:
        cleaned_text = cleaned_text.strip()
    return cleaned_text


def build_content_frame(frame: pd.DataFrame, text_columns: Sequence[str]) -> pd.DataFrame:
    require_columns(frame, ["id"], "Input frame")
    output_frame = frame.copy()
    content_series = pd.Series([""] * len(output_frame), index=output_frame.index, dtype="object")

    for column in text_columns:
        if column in output_frame.columns:
            column_text = output_frame[column].map(value_to_text)
        else:
            column_text = pd.Series([""] * len(output_frame), index=output_frame.index, dtype="object")
        content_series = content_series.str.cat(column_text, sep=" ")

    normalization = DEFAULT_CONFIG.normalization
    content_series = content_series.astype(str)
    if normalization.replace_separators:
        content_series = content_series.str.replace(_SEPARATOR_RE, " ", regex=True)
    if normalization.lowercase:
        content_series = content_series.str.lower()
    if normalization.collapse_whitespace:
        content_series = content_series.str.replace(_WHITESPACE_RE, " ", regex=True)
    if normalization.strip:
        content_series = content_series.str.strip()

    output_frame["content"] = content_series
    output_frame["id"] = output_frame["id"].astype(str)
    return output_frame


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def build_query_classifier_frame(query_frame: pd.DataFrame, include_tags: bool | None = None) -> pd.DataFrame:
    use_tags = DEFAULT_CONFIG.data_columns.use_query_tags_in_classifier if include_tags is None else include_tags
    columns = ["title", "text"]
    if use_tags and "tags" in query_frame.columns:
        columns.append("tags")
    return build_content_frame(query_frame, columns)
