# Retrieval Engine Competition Project

Information retrieval project for the Kaggle competition:
https://www.kaggle.com/competitions/retrieval-engine-competition

## Team
- Maksym DOLHOV
- Mehdi AGHAEI
- Nguyen Ho Bao KHANH
- Nima DAVARI

## Repository Structure
```text
retrieval_project/
├── data/                      # local competition data (ignored in git)
├── notebooks/
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
│   ├── evaluation.py          # Precision@K / Recall@K / MRR@K / MAP@K
│   └── utils.py               # data loading + Kaggle CSV writer
├── main.py                    # thin CLI entrypoint (calls src.pipeline)
├── PIPELINE.md                # workflow details
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

## Run Full Pipeline
```bash
python3 main.py
```

This command:
1. loads and preprocesses data,
2. evaluates TF-IDF and BM25 on training queries,
3. generates test-set submissions,
4. writes `outputs/solutions_SeaFour.csv` for Kaggle upload.

Select final upload model in `src/pipeline.py` with `FINAL_MODEL` (`"bm25"`, `"tfidf"`, or `"dense"`).
To include dense model comparison during training evaluation, set `RUN_DENSE_EVAL=True`.

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
