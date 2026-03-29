# Retrieval Engine Competition Project

Group: `SeaFour`  
Main notebook: `kaggle/kaggle-submission.ipynb`

This repository contains the final submission version of the retrieval project for the course competition.

## Final active system

The final pipeline uses:

- full-corpus document indexing on `title + text + tags`
- retrieval queries built from `title + text`
- Phase 1 comparison of `TF-IDF`, `BM25+`, and `Sentence-Transformer` embeddings
- a Phase 2 `TF-IDF + LinearSVC` category classifier
- soft category-aware reranking through a category bonus
- Kaggle submission export to `solutions_SeaFour.csv`

Latest confirmed public Kaggle score in this repo: `0.60007` from the corrected notebook submission

## Repository contents

- `kaggle/kaggle-submission.ipynb`: final notebook
- `PIPELINE.md`: compact description of the implemented workflow
- `reports/retrieval_project_report.tex`: final written report source
- `reports/retrieval_project_report.pdf`: compiled report
- `guide_pdf/retrieval_guide_rebuilt.tex`: teaching guide source
- `guide_pdf/retrieval_guide_rebuilt.pdf`: compiled teaching guide
- `requirements.txt`: Python dependencies for local execution

## Expected data files

The notebook expects the competition files in `data/`:

- `docs.json`
- `queries_train.json`
- `queries_test.json`
- `qgts_train.json`
- `submission.csv`

Important qrels note:

- `qgts_train.json` stores relevance under `relevant_doc_ids`, with nested `doc_id` entries
- the final notebook is aligned with that schema

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `umap-learn` is unavailable, the notebook falls back automatically to `t-SNE` for the 2D embedding projection.

## How to run

1. Make sure the competition files are available in `data/`.
2. Open `kaggle/kaggle-submission.ipynb`.
3. Run the notebook from top to bottom.
4. Review the outputs:
   - dataset summary tables
   - embedding shapes and 2D projection
   - Phase 1 retrieval comparison
   - classifier evaluation on queries and documents
   - Phase 2 reranking comparison
   - validation-selected model summary
5. Export the generated `solutions_SeaFour.csv` file for Kaggle.

## Runtime behavior

The notebook auto-detects the execution environment:

- Kaggle: `/kaggle/input/competitions/retrieval-engine-competition`
- Colab: a mounted Drive folder named `retrieval_project`
- Local: the nearest valid `data/` directory from the current working tree

## Key results

- Best Phase 1 validation model: `embedding` at `K=1000`
- Best Phase 2 validation model: `embedding + text_only classifier` at `K=1000`
- Best Phase 2 validation combined score: `0.57342`
- Latest public Kaggle score: `0.60007` from the corrected notebook submission

## Notes

- The notebook uses a corrected `70 / 30` train / validation split on labeled training queries and keeps `queries_test.json` as the only final test set.
- The classifier is evaluated separately on held-out documents and validation queries.
- Query classification is reported for both `text_only` and `text_plus_tags`, but the active submission pipeline keeps `text_only` as the brief-aligned default.
- Cross-encoder reranking is not part of the active submission path.
