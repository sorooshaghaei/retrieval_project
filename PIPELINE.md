# Retrieval Project Pipeline

## Goal

Build a retrieval engine for the competition with:

- Phase 1: compare `TF-IDF`, `BM25+`, and embedding retrieval
- Phase 2: add category classification and use it to improve ranking

## Inputs

The notebook expects the following files in `data/`:

| File | Role |
|---|---|
| `docs.json` | full document corpus |
| `queries_train.json` | labeled training queries |
| `queries_test.json` | Kaggle test queries |
| `qgts_train.json` | qrels for training queries |
| `submission.csv` | Kaggle submission template |

Qrels schema note:

- the final notebook reads relevance from `relevant_doc_ids`
- each relevant item contains a nested `doc_id`

## Preprocessing

The final notebook builds normalized `content` as follows:

- documents: `title + text + tags`
- retrieval queries: `title + text`
- primary classifier queries: `title + text`
- classifier ablation: `title + text + tags`

Normalization:

- lowercase
- replace `-`, `_`, `/` with spaces
- collapse repeated whitespace
- trim leading and trailing whitespace

Tokenization pattern:

```text
[a-z0-9]+
```

Lexical components use English stopword filtering.

## Full pipeline

### 1. Load and validate data

The notebook:

- loads documents, train queries, test queries, qrels, and the submission template
- checks required columns
- checks id uniqueness
- resolves runtime paths for local / Colab / Kaggle execution

### 2. Build task-specific text

The notebook constructs `content` fields for:

- retrieval documents
- retrieval queries
- classifier query variants

This keeps the active retrieval path aligned with the query text while still measuring the effect of query tags on classification.

### 3. Explore the dataset

The notebook reports:

- corpus sizes
- document and query length statistics
- category counts
- relevant-documents-per-query statistics

### 4. Build retrieval artifacts

The notebook prepares three retrieval methods:

1. TF-IDF
2. BM25+
3. embeddings with `all-MiniLM-L6-v2`

Artifacts are cached to avoid rebuilding large indexes and embeddings on every run.

### 5. Inspect embeddings

The notebook:

- prints document and query embedding shapes
- samples documents by category
- projects embeddings to 2D with `UMAP` or fallback `t-SNE`

### 6. Split labeled queries

Because `queries_test.json` is already the official unlabeled competition test set, the labeled training queries are split stratified by category only into:

- `70%` train
- `30%` validation

The classifier also uses a separate document holdout evaluation.

### 7. Phase 1 retrieval comparison

The validation split is used to compare the three retrieval models on:

- Recall@K
- Precision@K
- MRR@K
- runtime / latency

Best Phase 1 validation result:

- model: `embedding`
- `K = 1000`
- Recall: `0.85902`
- Precision: `0.00827`
- MRR: `0.50574`

### 8. Phase 2 category classification

The notebook trains a `TF-IDF + LinearSVC` classifier on:

- training documents
- split-train queries

It evaluates classification on:

- held-out documents
- validation queries
- `queries_test.json` for prediction only (no offline accuracy)

Two query variants are reported:

- `text_only` = `title + text`
- `text_plus_tags` = `title + text + tags`

The active submission path keeps `text_only` as the default variant.

### 9. Phase 2 classifier-aware reranking

For each retrieval model:

1. predict the query category
2. retrieve top-k candidate documents
3. min-max normalize scores row-wise
4. add a soft category bonus for category matches
5. rerank the candidate list

Best Phase 2 validation result:

- model: `embedding`
- classifier variant: `text_only`
- `K = 1000`
- Recall: `0.85902`
- Precision: `0.00827`
- MRR: `0.50720`
- Accuracy: `0.91919`
- Combined score: `0.57342`

### 10. Validation-selected export and Kaggle submission

The validation winner is exported directly to the unlabeled Kaggle queries after the corrected split change.

Submission export:

- export depth: `7500`
- output file: `solutions_SeaFour.csv`
- previous public Kaggle score: `0.60185` before the query-split correction

## Reproducibility notes

- document embeddings are cached
- lexical artifacts are cached
- classifier artifacts are cached
- category-preserving sampling is used for projection and downsampling
- the document corpus is indexed once and reused
- the active notebook is `kaggle/kaggle-submission.ipynb`
- cross-encoder reranking is not part of the active submission path
