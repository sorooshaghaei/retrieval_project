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

## Retrieval Components

| Component | Status | Role |
|-----------|--------|------|
| TF-IDF retriever | Implemented | Lexical baseline (`TfidfVectorizer`, cosine similarity) |
| BM25+ retriever | Implemented | Lexical baseline (`rank-bm25`) |
| Embedding retriever | Active | First-stage dense retrieval (`all-MiniLM-L6-v2`) |
| Query category classifier | Active | TF-IDF + LinearSVC category prediction |
| Cross-encoder reranker | Active | Second-stage reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`) |
| Strict dominant-category filter | Implemented, not active | Kept as optional utility |

## Active Execution Flow

1. Load data and validate schemas/ids.
2. Build normalized `content` fields.
3. Build/load cached artifacts:
- TF-IDF and BM25 indexes
- document/query embeddings
- category classifier
- cross-encoder model
4. Run first-stage retrieval (`embedding`) with `top_k=7500`.
5. Rerank top candidates with cross-encoder:
- `CROSS_ENCODER_RERANK_TOP_M = 30`
- soft category bonus applied when enabled
6. Evaluate on train queries with:
- Recall@K
- Precision@K
- MRR@K
- category accuracy
- combined score = average of the four metrics
7. Run on test queries and write Kaggle-format CSV.

## Key Configuration (Current Defaults)

| Parameter | Default |
|-----------|---------|
| `FINAL_MODEL` | `embedding` |
| `EVALUATION_MODELS` | `("embedding",)` |
| `EVALUATION_TOP_KS` | `[7500]` |
| `SUBMIT_TOP_K` | `7500` |
| `ENABLE_CROSS_ENCODER_RERANK` | `True` |
| `CROSS_ENCODER_RERANK_TOP_M` | `30` |
| `ENABLE_CATEGORY_FILTER` | `True` (used as soft bonus in reranking) |
| `DOMINANT_CATEGORY_TOP_N` | `20` (for category stats / optional strict filter logic) |

## How To Run

Open `kaggle/kaggle-submission.ipynb` and run all cells.

## Outputs

- Submission CSV: `solutions_SeaFour.csv`
- Updated report source: `reports/retrieval_project_report_updated.tex`
- Updated report PDF: `reports/retrieval_project_report_updated.pdf`
