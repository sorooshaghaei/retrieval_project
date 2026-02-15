# old codes are in the end of this file

# main.py runs the whole machine
"""
retrieval pipeline looks like this:
    - Load Data.
    - Preprocess (Make the 'content' column).
    - Run Model (algorithms).
    - Save Results.(submissions)
"""

import pandas as pd
import os
from src.utils import load_data
from src.preprocess import create_content_column
from src.evaluation import evaluate_retrieval

from src.models import run_tfidf_search, run_bm25_search, embed_retrieve

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
TOP_K = 10
SUBMISSION_K = 100
TEAM_NAME = "SeaFour"

import csv
import json


def write_kaggle_submission(results, sample_csv_path, output_csv_path):

    # 1) Build a mapping: query_id -> list of doc_ids (as strings)
    pred_map = {}
    for item in results:
        qid = str(item["query_id"])
        doc_list = item["relevant_docs"]

        doc_list_str = []
        for d in doc_list:
            doc_list_str.append(str(d))

        pred_map[qid] = doc_list_str

    # 2) Read the sample
    with open(sample_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = (
            reader.fieldnames
        )  # e.g. ["query_id", "relevant_doc_ids", "category"]

        sample_rows = []
        for row in reader:
            sample_rows.append(row)

    # Safety
    if fieldnames is None or len(fieldnames) < 2:
        raise ValueError("Sample submission CSV must have at least 2 columns.")

    id_col = fieldnames[0]
    pred_col = fieldnames[1]
    cat_col = fieldnames[2] if len(fieldnames) >= 3 else None

    # 3) Write output rows following the sample order
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in sample_rows:
            qid = str(row[id_col])

            if qid not in pred_map:
                raise ValueError("Missing prediction for query_id: " + qid)

            out_row = {}
            out_row[id_col] = qid

            # **IMPORTANT**: Kaggle expects JSON string
            out_row[pred_col] = json.dumps(pred_map[qid])

            # Keep the sample's category value if it exists, otherwise use "?"
            if cat_col is not None:
                sample_cat = row.get(cat_col, "?")
                if sample_cat == "" or sample_cat is None:
                    sample_cat = "?"
                out_row[cat_col] = sample_cat

            writer.writerow(out_row)


def run_pipeline():
    # 1. LOAD
    print("--- 1. Loading Data ---")
    docs, train_queries, test_queries, qrels_raw = load_data(DATA_DIR)

    # The qrels file is complex. We need a simple dictionary:
    # { "query_id": ["doc_id_1", "doc_id_2", ...] }
    print("Processing Ground Truth (Qrels)...")
    qrels = {}
    for qid, info in qrels_raw.items():
        # extract the list of relevant doc_ids from the inner dictionary
        qrels[str(qid)] = [str(item["doc_id"]) for item in info["relevant_doc_ids"]]

    # 2. PREPROCESS
    print("--- 2. Preprocessing ---")
    docs = create_content_column(docs, ["title", "text", "tags"])
    train_queries = create_content_column(train_queries, ["title", "text"])
    test_queries = create_content_column(test_queries, ["title", "text"])

    # 3. RUN & EVALUATE (TRAIN SET)
    print("\n--- TF-IDF EVALUATION ---")
    tfidf_results = run_tfidf_search(docs, train_queries, TOP_K)
    tfidf_metrics = evaluate_retrieval(tfidf_results, qrels, TOP_K)
    print(f"Precision@{TOP_K}: {tfidf_metrics['avg_precision']:.4f}")

    print("\n--- BM25 EVALUATION ---")
    bm25_results = run_bm25_search(docs, train_queries, TOP_K)
    bm25_metrics = evaluate_retrieval(bm25_results, qrels, TOP_K)
    print(f"Precision@{TOP_K}: {bm25_metrics['avg_precision']:.4f}")

    # uncomment if you want to test semantic search(it takes 1 hour to run!)
    # print("\n--- SEMANTIC SEARCH (DEEP LEARNING) ---")
    # # We will test on a tiny slice first to prevent freezing.
    # # LIMIT DATA FOR TESTING: Take only first 10 queries to test speed
    # semantic_results = embed_retrieve(docs, train_queries.head(10), TOP_K)

    # # must only evaluate the queries we actually ran
    # # slice the qrels dictionary to match our 10 queries
    # # qrels_subset = {k: qrels[k] for k in train_queries.head(10)["id"] if k in qrels}
    # ids_10 = train_queries.head(10)["id"].astype(str).tolist()
    # qrels_subset = {qid: qrels.get(qid, []) for qid in ids_10}

    # semantic_metrics = evaluate_retrieval(semantic_results, qrels_subset, TOP_K)
    # print(
    #     f"Precision@{TOP_K} (First 10 queries): {semantic_metrics['avg_precision']:.4f}"
    # )

    # 4. SUBMISSION (TEST SET)

    print("\n--- GENERATING SUBMISSIONS ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    template_path = os.path.join(DATA_DIR, "submission.csv")  # Kaggle template

    # bm25 submission (top 100)
    bm25_test = run_bm25_search(docs, test_queries, SUBMISSION_K)
    bm25_path = os.path.join(OUTPUT_DIR, f"solutions_{TEAM_NAME}_bm25.csv")
    write_kaggle_submission(bm25_test, template_path, bm25_path)
    print("-> saved:", bm25_path)

    # tf-idf submission (top 100) 
    tfidf_test = run_tfidf_search(docs, test_queries, SUBMISSION_K)
    tfidf_path = os.path.join(OUTPUT_DIR, f"solutions_{TEAM_NAME}_tfidf.csv")
    write_kaggle_submission(tfidf_test, template_path, tfidf_path)
    print("-> saved:", tfidf_path)

    # must be exactly solutions_SeaFour.csv
    # replace bellow "???" with final choosed algorith model !!!!!
    # upload_path = os.path.join(OUTPUT_DIR, f"solutions_{TEAM_NAME}.csv")
    # write_kaggle_submission( ??? , template_path, ???_path)
    # print("-->>>> upload-ready file:", upload_path)


    print("-> outputs/solutions_SeaFour.csv saved (Kaggle format)")
    print("Done!")

if __name__ == "__main__":
    run_pipeline()


# -----------old codes----------------------

# import pandas as pd
# import os
# import numpy as np
# from src.utils import load_data
# from src.evaluation import evaluate_retrieval
# from src.preprocess import create_content_column
# from rank_bm25 import BM25Plus
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# from nltk.tokenize import word_tokenize

# DATA_DIR = "data"
# OUTPUT_DIR = "outputs"
# TOP_K = 10


# def run_baseline():
#     # 1. LOAD DATA
#     print("--- 1. Loading Data ---")
#     docs_df, train_queries_df, test_queries_df, qrels = load_data(DATA_DIR)

#     # Process qrels (qgts_train.json) → list of doc_id strings
#     qrels_processed = {}
#     for qid, info in qrels.items():
#         qrels_processed[str(qid)] = [
#             str(item['doc_id'])
#             for item in info.get('relevant_doc_ids', [])
#         ]

#     # 2. PREPROCESS
#     print("--- 2. Preprocessing ---")
#     print("Processing Documents...")
#     docs_processed = create_content_column(docs_df, ["title", "text", "tags"])
#     docs_processed['id'] = docs_processed['id'].astype(str)
#     docs_processed['tokens'] = docs_processed['content'].fillna("").apply(
#         lambda txt: [t.lower() for t in word_tokenize(str(txt))]
#     )

#     print("Processing Queries...")
#     train_queries_processed = create_content_column(train_queries_df, ["title", "text"])
#     test_queries_processed = create_content_column(test_queries_df, ["title", "text"])

#     train_queries_processed['id'] = train_queries_processed['id'].astype(str)
#     test_queries_processed['id'] = test_queries_processed['id'].astype(str)

#     train_queries_processed['tokens'] = train_queries_processed['content'].fillna("").apply(
#         lambda txt: [t.lower() for t in word_tokenize(str(txt))]
#     )
#     test_queries_processed['tokens'] = test_queries_processed['content'].fillna("").apply(
#         lambda txt: [t.lower() for t in word_tokenize(str(txt))]
#     )

#     # 3. RETRIEVAL MODELS
#     print("--- 3. Running Retrieval Models ---")

#     # ------------------- BM25 -------------------
#     print("Training BM25...")
#     corpus = docs_processed['tokens'].tolist()
#     bm25 = BM25Plus(corpus)

#     # ------------------- TF-IDF -------------------
#     print("Training TF-IDF...")
#     vectorizer = TfidfVectorizer(lowercase=True)
#     doc_tfidf = vectorizer.fit_transform(docs_processed['content'].fillna(""))

#     # Containers
#     results_train_bm25 = []
#     results_test_bm25 = []
#     results_train_tfidf = []
#     results_test_tfidf = []

#     # ================== BM25 on Train ==================
#     for _, row in train_queries_processed.iterrows():
#         query_id = str(row["id"])
#         query_tokens = row['tokens']
#         doc_scores = bm25.get_scores(query_tokens)
#         topk_indices = np.argsort(doc_scores)[-TOP_K:][::-1]
#         top_docs = [str(d) for d in docs_processed["id"].iloc[topk_indices]]
#         results_train_bm25.append({"query_id": query_id, "relevant_docs": top_docs})

#     # ================== BM25 on Test ==================
#     for _, row in test_queries_processed.iterrows():
#         query_id = str(row["id"])
#         query_tokens = row['tokens']
#         doc_scores = bm25.get_scores(query_tokens)
#         topk_indices = np.argsort(doc_scores)[-TOP_K:][::-1]
#         top_docs = [str(d) for d in docs_processed["id"].iloc[topk_indices]]
#         results_test_bm25.append({"query_id": query_id, "relevant_docs": top_docs})

#     # ================== TF-IDF on Train ==================
#     for _, row in train_queries_processed.iterrows():
#         query_id = str(row["id"])
#         query_text = str(row['content']) if pd.notna(row['content']) else ""
#         query_vec = vectorizer.transform([query_text])
#         scores = cosine_similarity(query_vec, doc_tfidf).flatten()
#         topk_indices = np.argsort(scores)[-TOP_K:][::-1]
#         top_docs = [str(d) for d in docs_processed["id"].iloc[topk_indices]]
#         results_train_tfidf.append({"query_id": query_id, "relevant_docs": top_docs})

#     # ================== TF-IDF on Test ==================
#     for _, row in test_queries_processed.iterrows():
#         query_id = str(row["id"])
#         query_text = str(row['content']) if pd.notna(row['content']) else ""
#         query_vec = vectorizer.transform([query_text])
#         scores = cosine_similarity(query_vec, doc_tfidf).flatten()
#         topk_indices = np.argsort(scores)[-TOP_K:][::-1]
#         top_docs = [str(d) for d in docs_processed["id"].iloc[topk_indices]]
#         results_test_tfidf.append({"query_id": query_id, "relevant_docs": top_docs})

#     # 4. EVALUATION (on train set)
#     print("\n--- Evaluating BM25 ---")
#     metrics_bm25 = evaluate_retrieval(results_train_bm25, qrels_processed, TOP_K)
#     print(f"Average Recall@{TOP_K}:  {metrics_bm25['avg_recall']:.4f}")
#     print(f"Average Precision@{TOP_K}: {metrics_bm25['avg_precision']:.4f}")
#     print(f"MRR@{TOP_K}:           {metrics_bm25['mrr']:.4f}")

#     print("\n--- Evaluating TF-IDF ---")
#     metrics_tfidf = evaluate_retrieval(results_train_tfidf, qrels_processed, TOP_K)
#     print(f"Average Recall@{TOP_K}:  {metrics_tfidf['avg_recall']:.4f}")
#     print(f"Average Precision@{TOP_K}: {metrics_tfidf['avg_precision']:.4f}")
#     print(f"MRR@{TOP_K}:           {metrics_tfidf['mrr']:.4f}")

#     # 5. SAVE SUBMISSIONS
#     print("\n--- 5. Saving Submissions ---")
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     pd.DataFrame(results_test_bm25).to_csv(
#         os.path.join(OUTPUT_DIR, "submission_bm25.csv"), index=False
#     )
#     pd.DataFrame(results_test_tfidf).to_csv(
#         os.path.join(OUTPUT_DIR, "submission_tfidf.csv"), index=False
#     )

#     print(f"→ submission_bm25.csv saved")
#     print(f"→ submission_tfidf.csv saved")
#     print("Done!")


# if __name__ == "__main__":
#     run_baseline()
