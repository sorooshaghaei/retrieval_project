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

# helper formating function
def format_for_submission(results):
    df = pd.DataFrame(results)
    df["query_id"] = df["query_id"].astype(str)

    # Common Kaggle format for ranked doc ids:
    df["relevant_docs"] = df["relevant_docs"].apply(lambda xs: " ".join(map(str, xs)))
    return df

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

    print("\n--- SEMANTIC SEARCH (DEEP LEARNING) ---")
    # We will test on a tiny slice first to prevent freezing.
    # LIMIT DATA FOR TESTING: Take only first 100 queries to test speed
    semantic_results = embed_retrieve(docs, train_queries.head(100), TOP_K)

    # must only evaluate the queries we actually ran
    # slice the qrels dictionary to match our 100 queries
    # qrels_subset = {k: qrels[k] for k in train_queries.head(100)["id"] if k in qrels}
    ids_100 = train_queries.head(100)["id"].astype(str).tolist()
    qrels_subset = {qid: qrels.get(qid, []) for qid in ids_100}

    semantic_metrics = evaluate_retrieval(semantic_results, qrels_subset, TOP_K)
    print(
        f"Precision@{TOP_K} (First 100 queries): {semantic_metrics['avg_precision']:.4f}"
    )

    # 4. SUBMISSION (TEST SET)
    print("\n--- GENERATING SUBMISSION ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TF-IDF submission (on test)
    tfidf_test = run_tfidf_search(docs, test_queries, TOP_K)
    sub_tfidf = format_for_submission(tfidf_test)
    sub_tfidf.to_csv(os.path.join(OUTPUT_DIR, "submission_tfidf.csv"), index=False)


    # BM25 submission (on test)
    bm25_test = run_bm25_search(docs, test_queries, TOP_K)
    sub_bm25 = format_for_submission(bm25_test)
    sub_bm25.to_csv(os.path.join(OUTPUT_DIR, "submission_bm25.csv"), index=False)

    print(f"→ submission_bm25.csv saved")
    print(f"→ submission_tfidf.csv saved")
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
