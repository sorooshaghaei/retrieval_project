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
- `run_embedding_hybrid_search`: TF-IDF + BM25 candidate generation with sentence-transformer reranking

All models return a standard format:
```python
{"query_id": "...", "relevant_docs": ["doc1", "doc2", ...]}
```

## 4. Evaluation
Training queries are evaluated using `qgts_train.json` with:
- Precision@K
- Recall@K
- MRR@K
- MAP@K

Implementation is in `src/evaluation.py`.
`src/pipeline.py` (triggered by `main.py`) evaluates TF-IDF and BM25 by default, and can include hybrid embedding evaluation
by setting `RUN_EMBEDDING_HYBRID_EVAL=True`.

## 5. Submission Generation
For test queries, top-100 documents are produced, then exported to Kaggle format using `write_kaggle_submission` in `src/utils.py`.

Outputs are written to `outputs/`:
- `solutions_SeaFour_bm25.csv`
- `solutions_SeaFour_tfidf.csv`
- `solutions_SeaFour.csv` (final file to upload)

`solutions_SeaFour.csv` is generated from the model selected in `FINAL_MODEL` inside `src/pipeline.py`.

After writing the final file, the pipeline validates it against `data/submission.csv`
using `validate_submission_against_template`.

## 6. Run Command
```bash
python3 main.py
```

## 7. Submission Strategy
- Moodle submission: full project code + report PDF.
- Kaggle submission: use only `notebooks/kaggle/kaggle_submission.ipynb` to reproduce leaderboard CSV.
- Validation report: `notebooks/reports/submission_report.ipynb`.
- Phase-1 model comparison: `notebooks/phase1/phase1_retrieval_basics.ipynb`.
