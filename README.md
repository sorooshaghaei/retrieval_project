# Retrieval Engine Competition Project

**Group:** SeaFour  
**Project:** Kaggle Retrieval Engine Competition

## Team

- Maksym DOLHOV
- Mehdi AGHAEI
- Nguyen Ho Bao KHANH

## Overview

This repository now centers on a reusable Python pipeline in [`src/pipeline.py`](src/pipeline.py), not just a single notebook workflow.

The implemented system supports:

- first-stage retrieval with `tfidf`, `bm25`, or `embedding`
- category prediction with a TF-IDF + `LinearSVC` classifier
- optional category-filtered retrieval
- optional cross-encoder reranking
- leaderboard-style evaluation with Recall, Precision, MRR, and category Accuracy
- Kaggle submission writing based on the sample submission format

## Current Default Configuration

The defaults in [`src/config.py`](src/config.py) are:

- `final_model="embedding"`
- `evaluation_models=("embedding",)`
- `submit_top_k=7500`
- `enable_category_prediction=True`
- `enable_category_filter=True`
- `enable_cross_encoder_rerank=True`
- embedding model: `all-MiniLM-L6-v2`
- cross-encoder model: `cross-encoder/ms-marco-MiniLM-L6-v2`

So the out-of-the-box pipeline is: encode documents and queries with Sentence-Transformers, retrieve top documents by dense similarity, predict categories, and write a Kaggle submission. Category-filtered retrieval and cross-encoder reranking exist in the codebase but are disabled by default.

## Repository Structure

```text
retrieval_project/
├── data/                       # competition files (not committed)
├── cache/                      # cached indices, embeddings, classifiers, rerankers
├── notebooks/                  # experiments and submission notebooks
├── report/                     # LaTeX report and compiled PDF
├── src/
│   ├── pipeline.py             # main orchestration helpers
│   ├── config.py               # dataclass-based configuration
│   ├── data/                   # loading and text normalization
│   ├── retrieval/              # TF-IDF, BM25, embedding retrieval
│   ├── categorization/         # category classifier
│   ├── cross_encoder/          # reranker training and inference helpers
│   ├── evaluation/             # retrieval and leaderboard metrics
│   └── output/                 # Kaggle submission writing
├── PIPELINE.md                 # step-by-step pipeline description
├── README.md
└── requirements.txt
```

## Expected Input Files

Place these files in `data/`:

- `docs.json`
- `queries_train.json`
- `queries_test.json`
- `qgts_train.json`
- `submission.csv`

Required columns enforced by the code:

- documents: `id`, `title`, `text`, `tags`, `category`
- train queries: `id`, `title`, `text`, `tags`, `category`
- test queries: `id`, `title`, `text`, `tags`
- sample submission: `query_id`, `relevant_doc_ids`, `category`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How The Code Is Intended To Run

The pipeline is designed to be imported from notebooks or Python scripts.

Main stages exposed in [`src/pipeline.py`](src/pipeline.py):

1. `bootstrap()` resolves runtime paths and creates `cache/`
2. `load_project_frames()` loads JSON/CSV inputs and builds normalized `content`
3. `prepare_retrievers()` builds or loads retrieval artifacts
4. `predict_categories()` trains or loads the category classifier and predicts query categories
5. `build_cross_encoder_reranker()` optionally trains or loads the cross-encoder
6. `run_first_stage_retrieval()` runs dense or lexical retrieval
7. `rerank_retrieval_results()` optionally reranks retrieved documents
8. `write_submission()` writes the final Kaggle CSV

## Text Preparation

Text normalization in [`src/data/text.py`](src/data/text.py):

- concatenates configured fields into a single `content` column
- lowercases text
- replaces `-`, `_`, and `/` with spaces
- collapses repeated whitespace
- converts lists such as `tags` into space-joined text

By default:

- document retrieval text uses `title + text + tags`
- retrieval query text uses `title + text`
- category-classifier query text uses `title + text + tags`

## Evaluation

Implemented in [`src/evaluation/metrics.py`](src/evaluation/metrics.py):

- `Recall@k`
- `Precision@k`
- `MRR@k`
- category `Accuracy`
- `LeaderboardScore = 0.25 * (Recall + Precision + MRR + Accuracy)`

## Runtime Environments

[`src/infra/runtime.py`](src/infra/runtime.py) detects:

- local execution
- Kaggle
- Google Colab with Drive mounted

It resolves `data/`, `cache/`, and the output CSV automatically.

## Output

The default output file is `solutions_SeaFour.csv` in the project root.  
Cached artifacts are stored under `cache/`.

## Report

- source: [`report/retrieval_project_report.tex`](report/retrieval_project_report.tex)
- pdf: [`report/retrieval_project_report.pdf`](report/retrieval_project_report.pdf)
