# Retrieval Project Pipeline

## Inputs

The notebook expects these files in `data/`:

| File | Description |
|------|-------------|
| `docs.json` | 216,041 documents (`id`, `title`, `text`, `tags`, `category`) |
| `queries_train.json` | 327 training queries |
| `queries_test.json` | 141 test queries |
| `qgts_train.json` | Training relevance judgments |
| `submission.csv` | Sample submission template |

## Runtime Path Resolution

The notebook auto-detects runtime and resolves paths as follows:
- `kaggle`: `/kaggle/input/competitions/retrieval-engine-competition`
- `colab`: mounted Google Drive project folder
- `local`: nearest `data/` folder from current working directory

Output defaults to `solutions_SeaFour.csv` at the project root.

## Shared Preprocessing

All retrievers use normalized `content` text:
- Documents: `title + text + tags`
- Retrieval queries: `title + text`
- Classifier queries: `title + text + tags`

Normalization includes:
- lowercasing
- separator replacement for `-_/`
- whitespace collapsing
- trimming

Tokenization uses `TOKEN_PATTERN = [a-z0-9]+`.

For lexical components (TF-IDF, classifier TF-IDF, BM25 tokenization), English stopword filtering is enabled by default:
- `STOPWORD_FILTER_ENABLED = True`
- `STOPWORD_LANGUAGE = "english"`
- optional `CUSTOM_STOPWORDS`

The notebook now separates:
- fit-free normalization: lowercasing, separator cleanup, whitespace cleanup, and shared `content` construction
- learned preprocessing: the classifier TF-IDF vectorizer, which is fit only on the split-train query set and then reused unchanged on validation, hold-out test, and production queries

## Retrieval Components

| Component | Status | Role |
|-----------|--------|------|
| TF-IDF retriever | Implemented | Lexical baseline (`TfidfVectorizer`, cosine similarity) |
| BM25+ retriever | Implemented | Lexical baseline (`rank-bm25`) |
| Embedding retriever | Active | First-stage dense retrieval (`all-MiniLM-L6-v2`) |
| Query category classifier | Active | TF-IDF + LinearSVC category prediction fit on split-train queries |
| Cross-encoder reranker | Active | Second-stage reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`) |

## Active Execution Flow

1. Load data and validate schemas/ids.
2. Build normalized `content` fields.
3. Split the 327 labeled queries stratified into:
- 196 split-train queries
- 98 validation queries
- 33 hold-out test queries
4. Build/load cached document-side artifacts:
- TF-IDF and BM25 indexes
- document embeddings
- sentence-transformer weights
5. Fit the query category classifier on `split_train_queries_classifier_df` only.
6. Reuse that same fitted classifier transform to predict categories for:
- split-train queries
- validation queries
- hold-out test queries
- production test queries
7. Train/load the cross-encoder on `split_train_queries_df` and `train_ground_truth` only.
8. Run first-stage retrieval (`embedding`) on the validation queries with `top_k=7500`.
9. Rerank the validation results with the cross-encoder using:
- `rerank_top_m = 150`
- soft category bonus `= 2.0` when `ENABLE_CATEGORY_BOOST = True`
10. Score the validation ranking with:
- Recall@K
- Precision@K
- MRR@K
- category accuracy
- combined score = average of the four metrics
11. Keep the best validation configuration and run it once on the hold-out test split.
12. Run on production test queries and write the Kaggle-format CSV with the same train-fitted classifier and the same trained cross-encoder.

## Key Configuration (Current Defaults)

| Parameter | Default |
|-----------|---------|
| `FINAL_MODEL` | `embedding` |
| `EVALUATION_MODELS` | `("embedding",)` |
| `EVALUATION_TOP_KS` | `(7500,)` |
| `SUBMIT_TOP_K` | `7500` |
| `TRAIN_FRACTION` | `0.60` |
| `VALIDATION_FRACTION` | `0.30` |
| `TEST_FRACTION` | `0.10` |
| `ENABLE_CROSS_ENCODER_RERANK` | `True` |
| `CROSS_ENCODER_TRAIN_QUERY_LIMIT` | `327` |
| `CROSS_ENCODER_RERANK_TOP_M` | `150` |
| `CROSS_ENCODER_HARD_NEGATIVE_TOP_K` | `200` |
| `ENABLE_CATEGORY_BOOST` | `True` |
| `CATEGORY_MATCH_BONUS` | `2.0` |

## How To Run

Open `kaggle/kaggle-submissione.ipynb` and run all cells.

## Important Note

The current saved notebook is now leakage-safe with respect to the query-side fitted preprocessing and reranker: the classifier TF-IDF transform and the cross-encoder are fit only on the split-train queries, validation is used for selection, hold-out test is used once for offline estimation, and production test queries reuse the same fitted artifacts without refitting.

## Outputs

- Submission CSV: `solutions_SeaFour.csv`
- Updated report source: `reports/retrieval_project_report_updated.tex`
- Updated report PDF: `reports/retrieval_project_report_updated.pdf`
