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

All methods are evaluated on training queries (MAP@100, Recall@100), then the best model produces the test submission.

## Structure

```text
retrieval_project/
├── data/                              # competition data (not committed)
│   ├── docs.json                      # 216,041 documents
│   ├── queries_train.json             # 327 training queries
│   ├── queries_test.json              # 141 test queries
│   ├── qgts_train.json                # training ground truth
│   └── submission.csv                 # sample submission template
├── notebooks/
│   ├── kaggle-submission.ipynb        # active submission notebook
│   └── solutions_SeaFour.csv          # generated submission CSV
├── report/
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
2. Open `notebooks/kaggle-submission.ipynb`.
3. Run all cells.

The notebook auto-detects `/kaggle/input` first, then falls back to `../data`.

## Configuration

| Parameter | Default | Options |
|-----------|---------|--------|
| `FINAL_MODEL` | `hybrid` | `bm25`, `tfidf`, `embedding`, `hybrid` |
| `TOP_K` | `100` | documents per query |
| `HYBRID_BM25_CANDIDATES` | `500` | BM25+ candidates before re-ranking |
| `HYBRID_ALPHA` | `0.35` | BM25+ weight in score fusion |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformer model |

## Report

- Source: `report/retrieval_project_report.tex`
- PDF: `report/retrieval_project_report.pdf`

Build manually:
```bash
cd report && xelatex -interaction=nonstopmode retrieval_project_report.tex
```

With VS Code LaTeX Workshop, the PDF auto-builds on every save.
