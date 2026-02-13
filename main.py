# Runs the whole machine
# baseline.py codes are moved to here

"""
retrieval pipeline looks like this:
    - Load Data.
    - Preprocess (Make the 'content' column).
    - Run Model (algorithms).
    - Save Results.(submissions)
"""

import pandas as pd
import os
import numpy as np
from src.utils import load_data
from src.evaluation import evaluate_retrieval
from src.preprocess import create_content_column
from rank_bm25 import BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
TOP_K = 10


def run_baseline():
    # 1. LOAD DATA
    print("--- 1. Loading Data ---")
    docs_df, train_queries_df, test_queries_df, qrels = load_data(DATA_DIR)

    # Process qrels (qgts_train.json) → list of doc_id strings
    qrels_processed = {}
    for qid, info in qrels.items():
        qrels_processed[str(qid)] = [
            str(item['doc_id']) 
            for item in info.get('relevant_doc_ids', [])
        ]

    # 2. PREPROCESS
    print("--- 2. Preprocessing ---")
    print("Processing Documents...")
    docs_processed = create_content_column(docs_df, ["title", "text", "tags"])
    docs_processed['id'] = docs_processed['id'].astype(str)
    docs_processed['tokens'] = docs_processed['content'].fillna("").apply(
        lambda txt: [t.lower() for t in word_tokenize(str(txt))]
    )

    print("Processing Queries...")
    train_queries_processed = create_content_column(train_queries_df, ["title", "text"])
    test_queries_processed = create_content_column(test_queries_df, ["title", "text"])

    train_queries_processed['id'] = train_queries_processed['id'].astype(str)
    test_queries_processed['id'] = test_queries_processed['id'].astype(str)

    train_queries_processed['tokens'] = train_queries_processed['content'].fillna("").apply(
        lambda txt: [t.lower() for t in word_tokenize(str(txt))]
    )
    test_queries_processed['tokens'] = test_queries_processed['content'].fillna("").apply(
        lambda txt: [t.lower() for t in word_tokenize(str(txt))]
    )

    # 3. RETRIEVAL MODELS
    print("--- 3. Running Retrieval Models ---")

    # ------------------- BM25 -------------------
    print("Training BM25...")
    corpus = docs_processed['tokens'].tolist()
    bm25 = BM25Plus(corpus)

    # ------------------- TF-IDF -------------------
    print("Training TF-IDF...")
    vectorizer = TfidfVectorizer(lowercase=True)
    doc_tfidf = vectorizer.fit_transform(docs_processed['content'].fillna(""))

    # Containers
    results_train_bm25 = []
    results_test_bm25 = []
    results_train_tfidf = []
    results_test_tfidf = []

    # ================== BM25 on Train ==================
    for _, row in train_queries_processed.iterrows():
        query_id = str(row["id"])
        query_tokens = row['tokens']
        doc_scores = bm25.get_scores(query_tokens)
        topk_indices = np.argsort(doc_scores)[-TOP_K:][::-1]
        top_docs = [str(d) for d in docs_processed["id"].iloc[topk_indices]]
        results_train_bm25.append({"query_id": query_id, "relevant_docs": top_docs})

    # ================== BM25 on Test ==================
    for _, row in test_queries_processed.iterrows():
        query_id = str(row["id"])
        query_tokens = row['tokens']
        doc_scores = bm25.get_scores(query_tokens)
        topk_indices = np.argsort(doc_scores)[-TOP_K:][::-1]
        top_docs = [str(d) for d in docs_processed["id"].iloc[topk_indices]]
        results_test_bm25.append({"query_id": query_id, "relevant_docs": top_docs})

    # ================== TF-IDF on Train ==================
    for _, row in train_queries_processed.iterrows():
        query_id = str(row["id"])
        query_text = str(row['content']) if pd.notna(row['content']) else ""
        query_vec = vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, doc_tfidf).flatten()
        topk_indices = np.argsort(scores)[-TOP_K:][::-1]
        top_docs = [str(d) for d in docs_processed["id"].iloc[topk_indices]]
        results_train_tfidf.append({"query_id": query_id, "relevant_docs": top_docs})

    # ================== TF-IDF on Test ==================
    for _, row in test_queries_processed.iterrows():
        query_id = str(row["id"])
        query_text = str(row['content']) if pd.notna(row['content']) else ""
        query_vec = vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, doc_tfidf).flatten()
        topk_indices = np.argsort(scores)[-TOP_K:][::-1]
        top_docs = [str(d) for d in docs_processed["id"].iloc[topk_indices]]
        results_test_tfidf.append({"query_id": query_id, "relevant_docs": top_docs})

    # 4. EVALUATION (on train set)
    print("\n--- Evaluating BM25 ---")
    metrics_bm25 = evaluate_retrieval(results_train_bm25, qrels_processed, TOP_K)
    print(f"Average Recall@{TOP_K}:  {metrics_bm25['avg_recall']:.4f}")
    print(f"Average Precision@{TOP_K}: {metrics_bm25['avg_precision']:.4f}")
    print(f"MRR@{TOP_K}:           {metrics_bm25['mrr']:.4f}")

    print("\n--- Evaluating TF-IDF ---")
    metrics_tfidf = evaluate_retrieval(results_train_tfidf, qrels_processed, TOP_K)
    print(f"Average Recall@{TOP_K}:  {metrics_tfidf['avg_recall']:.4f}")
    print(f"Average Precision@{TOP_K}: {metrics_tfidf['avg_precision']:.4f}")
    print(f"MRR@{TOP_K}:           {metrics_tfidf['mrr']:.4f}")

    # 5. SAVE SUBMISSIONS
    print("\n--- 5. Saving Submissions ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pd.DataFrame(results_test_bm25).to_csv(
        os.path.join(OUTPUT_DIR, "submission_bm25.csv"), index=False
    )
    pd.DataFrame(results_test_tfidf).to_csv(
        os.path.join(OUTPUT_DIR, "submission_tfidf.csv"), index=False
    )

    print(f"→ submission_bm25.csv saved")
    print(f"→ submission_tfidf.csv saved")
    print("Done!")


if __name__ == "__main__":
    run_baseline()