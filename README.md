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
2. TF-IDF + LinearSVC category prediction for train/test queries
3. cross-encoder training on all 327 labeled train queries
4. offline evaluation on the same 327 train queries with cross-encoder reranking (`top_m=150`) and soft category bonus
5. final CSV export on test queries with submission-time reranking (`top_m=20`)

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
| `EVALUATION_TOP_KS` | `[7500]` |
| `SUBMIT_TOP_K` | `7500` |
| `ENABLE_CROSS_ENCODER_RERANK` | `True` |
| `CROSS_ENCODER_TRAIN_QUERY_LIMIT` | `327` |
| `CROSS_ENCODER_RERANK_TOP_M` | `150` (offline evaluation) |
| `Submission rerank_top_m` | `20` (hard-coded in submission cell) |
| `ENABLE_CATEGORY_FILTER` | `True` (soft bonus of `2.0` in reranking) |
| `CROSS_ENCODER_HARD_NEGATIVE_TOP_K` | `200` |
| `STOPWORD_FILTER_ENABLED` | `True` |
| `STOPWORD_LANGUAGE` | `"english"` |

## Outputs

- Submission CSV: `solutions_SeaFour.csv`
- Report source: `reports/retrieval_project_report_updated.tex`
- Report PDF: `reports/retrieval_project_report_updated.pdf`

## Notes

- A strict dominant-category filtering function is still implemented, but the active run uses soft category bonus inside cross-encoder reranking.
- The current notebook does not use a held-out validation split; its offline evaluation runs on `train_queries_df`.
- Cross-encoder training/inference is the most expensive stage, and caching is enabled for both the retriever artifacts and the cross-encoder model.
