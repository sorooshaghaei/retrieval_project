# Retrieval Engine Competition Project

**Group:** SeaFour — Kaggle Retrieval Engine Competition

## Team

- Maksym DOLHOV
- Mehdi AGHAEI
- Nguyen Ho Bao KHANH

## Overview

Notebook-first submission workflow implementing four retrieval methods:

| Method | Type | Description |
|--------|------|-------------|
| **TF-IDF** | Sparse lexical | Unigram+bigram vectoriser, cosine similarity |
| **BM25+** | Sparse lexical | Probabilistic ranking with length normalisation |
| **Embedding** | Dense semantic | Sentence-Transformer (`all-MiniLM-L6-v2`) |
| **Hybrid** | Sparse + Dense | BM25+ candidates → embedding re-ranking with score fusion |

The submission notebook focuses on generating the Kaggle CSV; offline evaluation (Precision/Recall/MRR/Accuracy) can be run separately if needed.

## Structure

```text
retrieval_project/
├── data/                              # competition data (not committed)
│   ├── docs.json                      # 216,041 documents
│   ├── queries_train.json             # 327 training queries
│   ├── queries_test.json              # 141 test queries
│   ├── qgts_train.json                # training ground truth
│   └── submission.csv                 # sample submission template
├── kaggle/
│   ├── kaggle_submission.ipynb        # submission notebook
│   └── solutions_SeaFour.csv          # generated submission CSV
├── reports/
│   ├── retrieval_project_report.tex   # LaTeX source (XeLaTeX)
│   └── retrieval_project_report.pdf   # compiled report
├── requirements.txt
├── PIPELINE.md                        # pipeline documentation
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

1. Place competition data files in `data/`.
2. Open `kaggle/kaggle_submission.ipynb`.
3. Run all cells.

The notebook auto-detects `/kaggle/input` first, then falls back to `../data`
(or `data/` if you launch Jupyter from the repo root).

## Configuration

| Parameter | Default | Options |
|-----------|---------|--------|
| `MODEL_NAME` | `bm25` | `bm25`, `tfidf`, `embedding_hybrid` |
| `TOP_K` | `100` | documents per query |
| `HYBRID_CANDIDATE_K` | `200` | candidates before embedding rerank |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformer model |
| `EMBEDDING_BATCH_SIZE` | `128` | embedding batch size |
| `OUTPUT_PATH` | `kaggle/solutions_SeaFour.csv` | output CSV location |

## Report

- Source: `reports/retrieval_project_report.tex`
- PDF: `reports/retrieval_project_report.pdf`

Build manually:
```bash
cd reports && xelatex -interaction=nonstopmode retrieval_project_report.tex
```

With VS Code LaTeX Workshop, the PDF auto-builds on every save.
