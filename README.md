# Retrieval Engine Competition Project

**Group:** SeaFour - Kaggle Retrieval Engine Competition

## Team

- Maksym DOLHOV
- Mehdi AGHAEI
- Nguyen Ho Bao KHANH

## Overview

This project is a notebook-first retrieval pipeline with:
- dense first-stage retrieval (`embedding`, Sentence-Transformer)
- optional lexical baselines (`tfidf`, `bm25`)
- TF-IDF + LinearSVC query category classifier
- cross-encoder second-stage reranking
- Kaggle CSV generation

The current active default path is:
1. `embedding` first-stage retrieval (`top_k=7500`)
2. stratified `60/30/10` split of the 327 labeled queries into `196 / 98 / 33`
3. TF-IDF + LinearSVC category-classifier fitting on the split-train queries only
4. cross-encoder training on the split-train queries only
5. validation-first model selection, then one hold-out test evaluation with cross-encoder reranking (`top_m=150`) and soft category bonus
6. final CSV export on production test queries using the same train-fitted artifacts

## Structure

```text
retrieval_project/
├── data/                                   # competition data (not committed)
│   ├── docs.json
│   ├── queries_train.json
│   ├── queries_test.json
│   ├── qgts_train.json
│   └── submission.csv
├── kaggle/
│   └── kaggle-submissione.ipynb            # main notebook
├── reports/
│   ├── retrieval_project_report_updated.tex
│   └── retrieval_project_report_updated.pdf
├── PIPELINE.md
├── requirements.txt
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

1. Put competition files in `data/`.
2. Open `kaggle/kaggle-submissione.ipynb`.
3. Run all cells.

Runtime path detection:
- Kaggle: `/kaggle/input/competitions/retrieval-engine-competition`
- Colab: mounted Google Drive project folder
- Local: nearest `data/` from current working directory

## Key Configuration (Notebook Defaults)

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
| `ENABLE_CATEGORY_BOOST` | `True` (soft bonus of `2.0` in reranking) |
| `CATEGORY_MATCH_BONUS` | `2.0` |
| `CROSS_ENCODER_HARD_NEGATIVE_TOP_K` | `200` |
| `STOPWORD_FILTER_ENABLED` | `True` |
| `STOPWORD_LANGUAGE` | `"english"` |

## Outputs

- Submission CSV: `solutions_SeaFour.csv`
- Report source: `reports/retrieval_project_report_updated.tex`
- Report PDF: `reports/retrieval_project_report_updated.pdf`

## Notes

- The notebook now uses a leakage-safe evaluation protocol: fit on split-train queries, select on validation, then report one hold-out test pass.
- Query-side fitted preprocessing is learned only on the split-train classifier frame and then reused with the same fitted transform on validation, hold-out test, and production queries.
- Cross-encoder training/inference is still the most expensive stage, and caching is enabled for both the retriever artifacts and the cross-encoder model.
