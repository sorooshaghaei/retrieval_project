from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

ModelName = Literal["tfidf", "bm25", "embedding"]


class RetrievalResult(TypedDict):
    query_id: str
    relevant_docs: list[str]


class GroundTruthEntry(TypedDict):
    relevant_doc_ids: set[str]
    total_relevant_docs: int
    category: str | None


@dataclass(frozen=True)
class RuntimePaths:
    runtime_env: Literal["colab", "kaggle", "local"]
    project_dir: Path
    work_dir: Path
    data_dir: Path
    cache_dir: Path
    output_path: Path
