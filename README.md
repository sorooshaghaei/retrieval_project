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
2. cross-encoder rerank of top candidates (`top_m=30`)
3. category-aware soft bonus during reranking
4. final CSV export

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
│   └── kaggle-submission.ipynb             # main notebook
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
2. Open `kaggle/kaggle-submission.ipynb`.
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
| `CROSS_ENCODER_RERANK_TOP_M` | `30` |
| `ENABLE_CATEGORY_FILTER` | `True` (soft bonus in reranking) |
| `STOPWORD_FILTER_ENABLED` | `True` |
| `STOPWORD_LANGUAGE` | `"english"` |

## Outputs

- Submission CSV: `solutions_SeaFour.csv`
- Report source: `reports/retrieval_project_report_updated.tex`
- Report PDF: `reports/retrieval_project_report_updated.pdf`

## Notes

- A strict dominant-category filtering function is still implemented, but the active run uses soft category bonus inside cross-encoder reranking.
- Cross-encoder training/inference is the most expensive stage; caching is enabled in the notebook.
