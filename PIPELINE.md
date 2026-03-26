# Retrieval Project Pipeline

## Inputs

The pipeline expects these files in `data/`:

| File | Required columns | Purpose |
|------|------------------|---------|
| `docs.json` | `id`, `title`, `text`, `tags`, `category` | document collection |
| `queries_train.json` | `id`, `title`, `text`, `tags`, `category` | training queries |
| `queries_test.json` | `id`, `title`, `text`, `tags` | test queries |
| `qgts_train.json` | ground-truth structure with `relevant_doc_ids`, `total_relevant_docs`, `category` | retrieval evaluation |
| `submission.csv` | `query_id`, `relevant_doc_ids`, `category` | submission schema template |

Duplicate `id` values are rejected for documents and queries.

## Runtime Resolution

[`src/infra/runtime.py`](src/infra/runtime.py) detects one of:

- local
- Kaggle
- Google Colab

It then resolves:

- `data_dir`
- `cache_dir`
- output CSV path

Default output filename: `solutions_SeaFour.csv`.

## Shared Preprocessing

[`src/data/text.py`](src/data/text.py) builds a normalized `content` field.

Normalization rules:

- lowercasing enabled
- `-`, `_`, `/` replaced by spaces
- repeated whitespace collapsed
- leading and trailing whitespace stripped
- lists such as `tags` converted to space-joined text

Configured source columns:

- documents for retrieval: `title`, `text`, `tags`
- train/test queries for retrieval: `title`, `text`
- train/test queries for category classification: `title`, `text`, `tags`

## Retrieval Backends

Implemented in [`src/retrieval/search.py`](src/retrieval/search.py) and [`src/retrieval/indexes.py`](src/retrieval/indexes.py).

Supported first-stage models:

| Model | Implementation | Notes |
|------|----------------|-------|
| `tfidf` | scikit-learn `TfidfVectorizer` | default n-grams `(1, 2)`, `min_df=2` with fallback to `1` |
| `bm25` | `rank_bm25.BM25Plus` | tokenized normalized text |
| `embedding` | Sentence-Transformers | default model `all-MiniLM-L6-v2` |

Current default first-stage model from [`src/config.py`](src/config.py): `embedding`.

## Category Prediction

Implemented in [`src/categorization/classifier.py`](src/categorization/classifier.py).

Behavior:

- trains or loads a TF-IDF + `LinearSVC` classifier on document `content` and `category`
- predicts categories for train and test queries
- can be evaluated against train query categories
- predicted categories can be written into the submission file

Default setting: `enable_category_prediction=True`.

## Optional Category-Filtered Retrieval

If `enable_category_filter=True`, the pipeline:

1. predicts a category for each query
2. restricts the searchable document pool to that category when possible
3. runs embedding retrieval inside the filtered subset
4. falls back to the full corpus if no documents exist for a predicted category

This path is implemented by `run_category_filtered_retrieval()`.

Default setting: disabled.

## Optional Cross-Encoder Reranking

Implemented in [`src/cross_encoder/training.py`](src/cross_encoder/training.py) and [`src/cross_encoder/inference.py`](src/cross_encoder/inference.py).

When enabled, the pipeline can:

1. build training pairs from query-document relevance labels
2. mine hard negatives with the embedding retriever
3. train or load a cached cross-encoder
4. rerank the top retrieved documents
5. apply an optional category bonus during reranking

Default reranker model: `cross-encoder/ms-marco-MiniLM-L6-v2`  
Default setting: `enable_cross_encoder_rerank=False`

## Caching

Artifacts are cached under `cache/`.

Cached components include:

- TF-IDF indices
- BM25 indices
- document embeddings
- query embeddings
- category classifier artifacts
- trained cross-encoder models

Caching is controlled through flags in [`src/config.py`](src/config.py).

## End-to-End Flow

The orchestration helpers in [`src/pipeline.py`](src/pipeline.py) follow this order:

1. `bootstrap()` resolves paths and creates cache directories
2. `load_project_frames()` loads raw inputs and builds normalized working frames
3. `prepare_retrievers()` prepares the configured retrieval model(s)
4. `predict_categories()` builds category predictions if enabled
5. `build_cross_encoder_reranker()` builds the reranker if enabled
6. `run_first_stage_retrieval()` retrieves top documents for train or test queries
7. `rerank_retrieval_results()` optionally reranks the retrieved lists
8. `write_submission()` writes Kaggle-formatted predictions

## Evaluation Metrics

[`src/evaluation/metrics.py`](src/evaluation/metrics.py) computes:

- `Recall@k`
- `Precision@k`
- `MRR@k`
- category `Accuracy`
- `LeaderboardScore`

`LeaderboardScore` is defined as the simple average of Recall, Precision, MRR, and Accuracy.

## Current Defaults Summary

From [`src/config.py`](src/config.py):

- `final_model="embedding"`
- `evaluation_models=("embedding",)`
- `evaluation_top_ks=(7500,)`
- `submit_top_k=7500`
- `enable_category_prediction=True`
- `enable_category_filter=True`
- `enable_cross_encoder_rerank=True`
