# Retrieval Engine Competition Project

**Group:** SeaFour  
**Project:** Kaggle Retrieval Engine Competition

## Team

- Maksym DOLHOV
- Mehdi AGHAEI
- Nguyen Ho Bao KHANH

## Overview

This repository contains the current retrieval pipeline, evaluation notebooks, submission notebooks, and the final LaTeX report for the course Kaggle retrieval project.

The implemented system supports:

- first-stage retrieval with `tfidf`, `bm25`, or `embedding`
- category prediction with a TF-IDF + `LinearSVC` classifier
- category-aware retrieval logic
- cross-encoder reranking
- offline leaderboard-style evaluation with Recall, Precision, MRR, and category Accuracy
- Kaggle submission writing from the sample submission schema

The project is no longer organized around a single notebook only. The reusable pipeline lives in [`src/pipeline.py`](src/pipeline.py), while the notebooks document analysis, evaluation, reranker experiments, and submission generation.

## Current Project State

The current defaults in [`src/config.py`](src/config.py) are:

- `final_model="embedding"`
- `evaluation_models=("embedding",)`
- `evaluation_top_ks=(12500,)`
- `submit_top_k=12500`
- `enable_category_prediction=True`
- `enable_category_filter=True`
- `enable_cross_encoder_rerank=True`
- embedding model: `all-MiniLM-L6-v2`
- cross-encoder model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- rerank depth: `45`
- category bonus: `0.5`
- classifier query tags enabled: `True`

So the current pipeline direction is:

1. build normalized document and query text
2. retrieve candidates with embeddings by default
3. predict query categories
4. apply category-aware retrieval logic
5. rerank the head of the candidate list with a cross-encoder
6. export `solutions_SeaFour.csv`

Important project note: the repository distinguishes between saved offline evaluation outputs, current code defaults, and external Kaggle leaderboard results. The report in [`reports/retrieval_project_report.tex`](reports/retrieval_project_report.tex) documents those differences explicitly.

## Repository Structure

```text
retrieval_project/
├── data/                                  # competition files (not committed)
├── cache/                                 # cached indices, embeddings, classifiers, rerankers
├── notebooks/
│   ├── data-analysis.ipynb                # dataset exploration
│   ├── retrieval-evaluation.ipynb         # offline retrieval / Stage 2 evaluation
│   ├── reranker-evaluation.ipynb          # reranker-focused experiments
│   ├── kaggle-submission.ipynb            # main submission notebook
│   └── kaggle-submission-standalone.ipynb # standalone submission variant
├── reports/
│   ├── retrieval_project_report.tex       # canonical LaTeX report source
│   ├── retrieval_project_report.pdf       # compiled report
│   └── figures/embedding_projection.png   # report figure asset
├── src/
│   ├── pipeline.py                        # orchestration helpers
│   ├── config.py                          # dataclass-based configuration
│   ├── data/                              # loading and text normalization
│   ├── retrieval/                         # TF-IDF, BM25, embedding retrieval
│   ├── categorization/                    # category classifier
│   ├── cross_encoder/                     # cross-encoder training / inference
│   ├── reranking/                         # reranking logic
│   ├── evaluation/                        # retrieval and leaderboard metrics
│   ├── output/                            # diagnostics and submission writing
│   ├── infra/                             # runtime and notebook helpers
│   └── cache/                             # cache store helpers
├── PIPELINE.md
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

Required columns enforced by the pipeline:

- documents: `id`, `title`, `text`, `tags`, `category`
- train queries: `id`, `title`, `text`, `tags`, `category`
- test queries: `id`, `title`, `text`, `tags`
- sample submission: `query_id`, `relevant_doc_ids`, `category`

The qrels file is expected to provide `relevant_doc_ids` entries compatible with the loader used in the notebooks and report.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Main Pipeline API

The main orchestration helpers are defined in [`src/pipeline.py`](src/pipeline.py):

1. `bootstrap()` resolves runtime paths and creates the cache directory
2. `load_project_frames()` loads raw inputs and builds normalized working frames
3. `prepare_retrievers()` prepares the configured retrieval backends
4. `predict_categories()` trains or loads the classifier and predicts query categories
5. `build_cross_encoder_reranker()` trains or loads the reranker when enabled
6. `run_first_stage_retrieval()` runs retrieval for train or test queries
7. `rerank_retrieval_results()` reranks the retrieved head when enabled
8. `write_submission()` writes the Kaggle-formatted CSV

## Text Preparation

Normalization is implemented in [`src/data/text.py`](src/data/text.py).

The project builds a normalized `content` field by:

- lowercasing text
- replacing `-`, `_`, and `/` with spaces
- collapsing repeated whitespace
- trimming leading and trailing whitespace
- converting list fields such as `tags` into space-joined text

Configured source columns:

- document retrieval text: `title + text + tags`
- retrieval query text: `title + text`
- classifier query text: `title + text + tags`

## Evaluation and Submission Workflow

The notebooks currently play different roles:

- [`notebooks/data-analysis.ipynb`](notebooks/data-analysis.ipynb): dataset inspection and exploratory analysis
- [`notebooks/retrieval-evaluation.ipynb`](notebooks/retrieval-evaluation.ipynb): offline evaluation of retrieval and Stage 2 variants
- [`notebooks/reranker-evaluation.ipynb`](notebooks/reranker-evaluation.ipynb): reranker-specific experiments
- [`notebooks/kaggle-submission.ipynb`](notebooks/kaggle-submission.ipynb): main submission workflow
- [`notebooks/kaggle-submission-standalone.ipynb`](notebooks/kaggle-submission-standalone.ipynb): standalone submission version

Offline metrics are computed in [`src/evaluation/metrics.py`](src/evaluation/metrics.py):

- `Recall@k`
- `Precision@k`
- `MRR@k`
- category `Accuracy`
- `LeaderboardScore = 0.25 * (Recall + Precision + MRR + Accuracy)`

## Runtime Environments

[`src/infra/runtime.py`](src/infra/runtime.py) detects:

- local execution
- Kaggle
- Google Colab

It resolves `data/`, `cache/`, and the output CSV path automatically.

## Outputs

- default submission file: `solutions_SeaFour.csv`
- cached artifacts: `cache/`
- canonical report source: [`reports/retrieval_project_report.tex`](reports/retrieval_project_report.tex)
- compiled report: [`reports/retrieval_project_report.pdf`](reports/retrieval_project_report.pdf)

## Additional Documentation

- [`PIPELINE.md`](PIPELINE.md): step-by-step pipeline description
- [`reports/retrieval_project_report.tex`](reports/retrieval_project_report.tex): final written report
