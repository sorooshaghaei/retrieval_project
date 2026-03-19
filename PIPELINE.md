# Retrieval Project Pipeline

## Inputs

The pipeline expects these files in `data/`:

| File | Description |
|------|-------------|
| `docs.json` | 216,041 documents (id, text, title, tags, category) |
| `queries_train.json` | 327 training queries |
| `queries_test.json` | 141 test queries |
| `qgts_train.json` | Training ground-truth relevance judgements |
| `submission.csv` | Sample submission template |

## Shared Preprocessing

Every model receives one normalised text field (`content`):
- **Documents:** `title + text + tags` (tags are space-joined)
- **Queries:** `title + text`

All text is lowercased. Missing values, lists, and mixed types are handled by `value_to_text()`.

## Retrieval Methods

| Method | Type | Description |
|--------|------|-------------|
| TF-IDF | Sparse lexical | Unigram+bigram TfidfVectorizer, cosine similarity |
| BM25+ | Sparse lexical | Tokenised corpus with `rank-bm25`, length normalisation |
| Embedding | Dense semantic | Sentence-Transformer (`all-MiniLM-L6-v2`), dot product |
| Hybrid | Sparse + Dense | BM25+ retrieves top-500 candidates → embedding re-ranking with score fusion |

## Execution Flow

1. **Environment setup** — auto-detect Kaggle or local `data/` directory.
2. **Preprocessing** — build shared `content` column for docs and queries.
3. **Retrieval** — run the selected model (`bm25`, `tfidf`, or `embedding_hybrid`) on test queries.
4. **CSV writing** — serialise results in Kaggle format.
5. **Preview** — display the first rows of the output CSV.

## How to Run

Open `kaggle/kaggle_submission.ipynb` and run all cells.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `bm25` | `bm25`, `tfidf`, or `embedding_hybrid` |
| `TOP_K` | `100` | Documents per query |
| `HYBRID_CANDIDATE_K` | `200` | Candidates before embedding rerank |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformer model |
| `EMBEDDING_BATCH_SIZE` | `128` | Encoding batch size |
| `OUTPUT_PATH` | `kaggle/solutions_SeaFour.csv` | Output CSV path |

## Output

- **Submission CSV:** `kaggle/solutions_SeaFour.csv`
- **Report:** `reports/retrieval_project_report.pdf`
