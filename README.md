# Retrieval Engine Competition Project

Information retrieval project for the Kaggle competition:
https://www.kaggle.com/competitions/retrieval-engine-competition

## Team
- Maksym DOLHOV
- Mehdi AGHAEI
- Nguyen Ho Bao KHANH

<<<<<<< Updated upstream
## Repository Structure
=======
## Overview

Notebook-first submission workflow implementing the three required Phase 1 retrieval methods:

| Method | Type | Description |
|--------|------|-------------|
| **TF-IDF** | Sparse lexical | Unigram+bigram vectoriser, cosine similarity |
| **BM25+** | Sparse lexical | Probabilistic ranking with length normalisation |
| **Embedding** | Dense semantic | Sentence-Transformer (`all-MiniLM-L6-v2`) |

All required methods are evaluated on training queries with `Precision@k`, `Recall@k`, `MRR@k`, and `Accuracy`. In the current stage-0 setup, `Accuracy` is reported as `0.0`. The final Kaggle submission is produced with the embedding model.

## Structure

>>>>>>> Stashed changes
```text
retrieval_project/
├── data/                      # local competition data (ignored in git)
├── notebooks/
<<<<<<< Updated upstream
│   ├── kaggle/
│   │   └── kaggle_submission.ipynb
│   ├── phase1/
│   │   └── phase1_retrieval_basics.ipynb
│   ├── explain/
│   │   ├── retrieval_explained.ipynb
│   │   └── model_implementations.ipynb
│   ├── reports/
│   │   └── submission_report.ipynb
│   ├── analysis/
│   │   └── basic_analysis.ipynb
│   └── assets/
│       └── no_text_problem.png
├── outputs/                   # generated submissions/metrics (ignored in git)
├── src/
│   ├── pipeline.py            # end-to-end orchestration
│   ├── preprocess.py          # content construction + normalization
│   ├── models.py              # TF-IDF / BM25 / Dense retrieval
│   ├── evaluation.py          # Precision@K / Recall@K / MRR@K / Accuracy
│   └── utils.py               # data loading + Kaggle CSV writer
├── main.py                    # thin CLI entrypoint (calls src.pipeline)
├── PIPELINE.md                # workflow details
=======
│   ├── kaggle-submission.ipynb        # Phase 1 notebook
│   └── solutions_SeaFour.csv          # generated submission CSV
├── report/
│   ├── retrieval_project_report.tex   # LaTeX source (XeLaTeX)
│   └── retrieval_project_report.pdf   # compiled report
>>>>>>> Stashed changes
├── requirements.txt
└── README.md
```

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required Data Files
Put these in `data/`:
- `docs.json`
- `queries_train.json`
- `queries_test.json`
- `qgts_train.json`
- `submission.csv`

<<<<<<< Updated upstream
## Run Full Pipeline
=======
1. Place competition data files in `data/`.
2. Open `notebooks/kaggle-submission.ipynb`.
3. Run all cells.

The notebook auto-detects `/kaggle/input` first, then falls back to `../data`.

## Configuration

| Parameter | Default | Options |
|-----------|---------|--------|
| `FINAL_MODEL` | `embedding` | `embedding` |
| `TOP_K` | `100` | documents per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformer model |

## Report

- Source: `report/retrieval_project_report.tex`
- PDF: `report/retrieval_project_report.pdf`

Build manually:
>>>>>>> Stashed changes
```bash
python3 main.py
```

This command:
1. loads and preprocesses data,
2. evaluates TF-IDF and BM25 on training queries,
3. generates test-set submissions,
4. writes `outputs/solutions_SeaFour.csv` for Kaggle upload.

Select final upload model in `src/pipeline.py` with `FINAL_MODEL` (`"bm25"` or `"tfidf"`).

## Notebooks
- `notebooks/kaggle/kaggle_submission.ipynb`:
  Minimal, reproducible notebook for Kaggle submission generation only.
- `notebooks/explain/retrieval_explained.ipynb`:
  Explanation notebook describing data flow, ranking logic, and why models behave differently.
- `notebooks/explain/model_implementations.ipynb`:
  Model mechanics from scratch so retrieval methods are transparent and not black boxes.
- `notebooks/phase1/phase1_retrieval_basics.ipynb`:
  Phase-1 benchmark notebook comparing TF-IDF, BM25+, and embedding retrieval with standard IR metrics.
- `notebooks/reports/submission_report.ipynb`:
  Validation report notebook that checks `solutions_SeaFour.csv` against `data/submission.csv`.

## Submission Guidance
- Moodle: submit full codebase + report PDF.
- Kaggle: submit only `notebooks/kaggle/kaggle_submission.ipynb` (or an equivalent minimal retrieval-to-CSV notebook).
