# Retrieval Project Pipeline

## Inputs
The pipeline expects these files in `data/`:
- `docs.json`
- `queries_train.json`
- `queries_test.json`
- `qgts_train.json`
- `submission.csv`

## Shared Preprocessing
Every model receives one normalized text field:
- documents: `title + text + tags`
- queries: `title + text`

The transformation happens in [src/preprocess.py](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/src/preprocess.py).

## Runtime Configuration
Pipeline settings are centralized in [src/config.py](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/src/config.py).

Important options:
- `eval_models`
- `submission_models`
- `final_model`
- `run_embedding_hybrid_eval`
- `embedding_hybrid_eval_query_limit`

## Execution Flow
1. Load raw competition data with [src/utils.py](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/src/utils.py).
2. Build the `content` column with [src/preprocess.py](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/src/preprocess.py).
3. Evaluate train queries with [src/evaluation.py](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/src/evaluation.py).
4. Generate Kaggle CSVs with [src/models.py](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/src/models.py) and [src/utils.py](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/src/utils.py).
5. Validate the final file against the sample submission template.

## Main Command
```bash
python3 main.py
```

## Learning Material
- Workbook: [notebooks/retrieval_project_workbook.ipynb](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/notebooks/retrieval_project_workbook.ipynb)
- Report: [report/retrieval_project_report.pdf](/Users/sorooshaghaei/Desktop/Paris_cite_projects/retrieval_project/report/retrieval_project_report.pdf)
