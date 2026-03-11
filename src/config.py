"""Central configuration for the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Project-wide runtime settings for evaluation and submission generation."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    team_name: str = "SeaFour"

    eval_k: int = 10
    submission_k: int = 100

    eval_models: tuple[str, ...] = ("tfidf", "bm25")
    submission_models: tuple[str, ...] = ("tfidf", "bm25", "embedding_hybrid")
    final_model: str = "embedding_hybrid"

    run_embedding_hybrid_eval: bool = True
    embedding_hybrid_eval_query_limit: int = 100


DEFAULT_CONFIG = PipelineConfig()
