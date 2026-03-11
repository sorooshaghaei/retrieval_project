# Retrieval Engine Competition Project

Information retrieval project for the Kaggle competition:
https://www.kaggle.com/competitions/retrieval-engine-competition

## Team
- Maksym DOLHOV
- Mehdi AGHAEI
- Nguyen Ho Bao KHANH

## Clean Structure
```text
retrieval_project/
├── data/                                 # local competition data (ignored in git)
├── notebooks/
│   └── retrieval_project_workbook.ipynb  # one visual, teaching-first notebook
├── outputs/                              # generated submissions (ignored in git)
├── report/
│   ├── retrieval_project_report.tex      # A4 LaTeX source
│   └── retrieval_project_report.pdf      # compiled report
├── src/
│   ├── config.py                         # runtime configuration
│   ├── pipeline.py                       # orchestration
│   ├── preprocess.py                     # shared text normalization
│   ├── models.py                         # TF-IDF, BM25, dense, hybrid
│   ├── evaluation.py                     # Precision@K, Recall@K, MRR, MAP
│   └── utils.py                          # data loading + CSV validation/writing
├── main.py                               # thin CLI entrypoint
├── PIPELINE.md                           # short workflow reference
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
Put these files in `data/`:
- `docs.json`
- `queries_train.json`
- `queries_test.json`
- `qgts_train.json`
- `submission.csv`

## Run The Project
```bash
python3 main.py
```

The pipeline:
1. loads the competition files,
2. builds one normalized `content` field for docs and queries,
3. evaluates the configured retrieval models on train queries,
4. writes Kaggle-ready CSV files to `outputs/`,
5. validates the final submission against the sample template.

Runtime settings now live in [src/config.py](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/src/config.py). The main knobs are:
- `final_model`
- `eval_models`
- `submission_models`
- `run_embedding_hybrid_eval`
- `embedding_hybrid_eval_query_limit`

## Learning Artifacts
- [notebooks/retrieval_project_workbook.ipynb](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/notebooks/retrieval_project_workbook.ipynb): one consolidated notebook with data exploration, plots, model intuition, guided exercises, and code walk-throughs.
- [report/retrieval_project_report.tex](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/report/retrieval_project_report.tex): A4 report source.
- [report/retrieval_project_report.pdf](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/report/retrieval_project_report.pdf): compiled report with appendix and study material.

## Build The PDF
```bash
cd report
xelatex -interaction=nonstopmode retrieval_project_report.tex
```

## Notes
- The workbook is designed for learning, so it includes visual explanations, active-learning prompts, and code commentary.
- The codebase keeps the executable pipeline in `src/` and the teaching material in `notebooks/` and `report/`, so experimentation and submission logic stay separated.
