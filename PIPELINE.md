# Retrieval Project Pipeline

## 1. Data Inputs
The pipeline expects the following files inside `data/`:
- `docs.json`
- `queries_train.json`
- `queries_test.json`
- `qgts_train.json`
- `submission.csv` (Kaggle template)

## 2. Preprocessing
We create a unified `content` field for each row:
- Documents: `title + text + tags`
- Queries: `title + text`

Each field is lowercased and normalized into plain strings so all retrieval models consume the same input.

## 3. Retrieval Models
Implemented in `src/models.py`:
- `run_tfidf_search`: TF-IDF vectors + cosine similarity
- `run_bm25_search`: BM25+ lexical ranking

<<<<<<< Updated upstream
All models return a standard format:
```python
{"query_id": "...", "relevant_docs": ["doc1", "doc2", ...]}
```
=======
Every model receives one normalised text field (`content`):
- **Documents:** `title + text + tags` (tags are space-joined)
- **Queries:** `title + text + tags`
>>>>>>> Stashed changes

## 4. Evaluation
Training queries are evaluated using `qgts_train.json` with:
- Precision@K
- Recall@K
- MRR@K
- Accuracy

Implementation is in `src/evaluation.py`.
`src/pipeline.py` (triggered by `main.py`) evaluates TF-IDF and BM25 by default.

<<<<<<< Updated upstream
## 5. Submission Generation
For test queries, top-100 documents are produced, then exported to Kaggle format using `write_kaggle_submission` in `src/utils.py`.
=======
| Method | Type | Description |
|--------|------|-------------|
| TF-IDF | Sparse lexical | Unigram+bigram TfidfVectorizer, cosine similarity |
| BM25+ | Sparse lexical | Tokenised corpus with `rank-bm25`, length normalisation |
| Embedding | Dense semantic | Sentence-Transformer (`all-MiniLM-L6-v2`), dot product |
>>>>>>> Stashed changes

Outputs are written to `outputs/`:
- `solutions_SeaFour_bm25.csv`
- `solutions_SeaFour_tfidf.csv`
- `solutions_SeaFour.csv` (final file to upload)

<<<<<<< Updated upstream
`solutions_SeaFour.csv` is generated from the model selected in `FINAL_MODEL` inside `src/pipeline.py`.
=======
1. **Environment setup** — auto-detect Kaggle or local `data/` directory.
2. **Preprocessing** — build shared `content` column for docs and queries.
3. **Dataset analysis** — inspect fields, counts, lengths, categories, and relevance-judgement distributions.
4. **Evaluation** — run TF-IDF, BM25+, and embedding retrieval on 327 training queries; compute `Precision@k`, `Recall@k`, `MRR@k`, and `Accuracy`. In the current stage-0 setup, `Accuracy = 0.0`.
5. **Embedding inspection** — print embedding shapes and project them in 2D with t-SNE.
6. **Test retrieval** — run the embedding model on 141 test queries.
7. **CSV writing** — serialise results in Kaggle format.
8. **Preview** — display the first rows of the output CSV.
>>>>>>> Stashed changes

After writing the final file, the pipeline validates it against `data/submission.csv`
using `validate_submission_against_template`.

## 6. Run Command
```bash
python3 main.py
```

<<<<<<< Updated upstream
## 7. Submission Strategy
- Moodle submission: full project code + report PDF.
- Kaggle submission: use only `notebooks/kaggle/kaggle_submission.ipynb` to reproduce leaderboard CSV.
- Validation report: `notebooks/reports/submission_report.ipynb`.
- Phase-1 model comparison: `notebooks/phase1/phase1_retrieval_basics.ipynb`.
=======
## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FINAL_MODEL` | `embedding` | Final submission model |
| `TOP_K` | `100` | Documents per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformer model |

## Output

- **Submission CSV:** `notebooks/solutions_SeaFour.csv`
- **Report:** `report/retrieval_project_report.pdf`
>>>>>>> Stashed changes
