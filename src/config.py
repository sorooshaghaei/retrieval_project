from __future__ import annotations

from dataclasses import dataclass, field

from .types import ModelName


@dataclass(frozen=True)
class NormalizationConfig:
    lowercase: bool = True
    replace_separators: bool = True
    separator_chars: str = "-_/"
    collapse_whitespace: bool = True
    strip: bool = True


@dataclass(frozen=True)
class TFIDFConfig:
    lowercase: bool = True
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    delta: float = 1.0


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 256
    query_chunk_size: int = 32


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    epochs: int = 5
    batch_size: int = 32
    infer_batch_size: int = 64
    max_length: int = 256
    max_positives_per_query: int = 4
    negatives_per_positive: int = 4
    hard_negative_top_k: int = 200
    train_query_limit: int = 327
    rerank_top_m: int = 30
    zscore_weight: float = 0.75
    category_bonus: float = 0.5
    fp16: bool = True
    random_seed: int = 42
    enable_cache: bool = True


@dataclass(frozen=True)
class RetrievalPipelineConfig:
    final_model: ModelName = "embedding"
    evaluation_models: tuple[ModelName, ...] = ("embedding",)
    evaluation_top_ks: tuple[int, ...] = (7_500,)
    submit_top_k: int = 7_500
    enable_category_prediction: bool = True
    enable_category_filter: bool = True
    enable_cross_encoder_rerank: bool = True
    enable_embedding_cache: bool = True
    enable_classic_cache: bool = True
    token_pattern: str = r"[a-z0-9]+"


@dataclass(frozen=True)
class DataColumns:
    document_text_columns: tuple[str, ...] = ("title", "text", "tags")
    retrieval_query_columns: tuple[str, ...] = ("title", "text")
    classifier_query_columns: tuple[str, ...] = ("title", "text", "tags")
    use_query_tags_in_classifier: bool = True


@dataclass(frozen=True)
class DiagnosticsConfig:
    top_category_stats_k: int = 20
    query_category_stats_limit: int = 5


@dataclass(frozen=True)
class AllConfig:
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    retrieval_pipeline: RetrievalPipelineConfig = field(default_factory=RetrievalPipelineConfig)
    data_columns: DataColumns = field(default_factory=DataColumns)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


DEFAULT_CONFIG = AllConfig()
