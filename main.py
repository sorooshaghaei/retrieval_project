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
from src.utils import load_data
from src.evaluation import evaluate_retrieval
from src.preprocess import create_content_column
from rank_bm25 import BM25Plus
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

import numpy as np
DATA_DIR = "data"  # Folder where json files are
OUTPUT_DIR = "outputs"  # Folder where submission.csv goes
TOP_K = 10  # How many docs to retrieve per query


def run_baseline():
    # 1. LOAD DATA
    # -----------------------------------------------------
    print("--- 1. Loading Data ---")
    docs_df, train_queries_df, test_queries_df, qrels = load_data(DATA_DIR)

    # Process qrels to extract list of doc_ids
    qrels_processed = {}
    for qid, info in qrels.items():
        qrels_processed[str(qid)] = [str(item['doc_id']) for item in info.get('relevant_doc_ids', [])]

    # 2. PREPROCESS
    # -----------------------------------------------------
    print("--- 2. Preprocessing ---")
    # For Documents: Merge title + text + tags
    print("Processing Documents...")
    docs_processed = create_content_column(docs_df, ["title", "text", "tags"])

    # Ensure doc id is string (normalize)
    docs_processed['id'] = docs_processed['id'].astype(str)

    # Lowercase and tokenize content
    docs_processed['tokens'] = docs_processed['content'].fillna("").apply(
        lambda txt: [t.lower() for t in word_tokenize(str(txt))]
    )

    # For Queries: Merge title + text (queries might not have tags, check first)
    print("Processing Queries...")
    train_queries_processed = create_content_column(train_queries_df, ["title", "text"])
    test_queries_processed = create_content_column(test_queries_df, ["title", "text"])

    # Normalize query ids to string
    train_queries_processed['id'] = train_queries_processed['id'].astype(str)
    test_queries_processed['id'] = test_queries_processed['id'].astype(str)

    # Lowercase + tokenize queries
    train_queries_processed['tokens'] = train_queries_processed['content'].fillna("").apply(
        lambda txt: [t.lower() for t in word_tokenize(str(txt))]
    )
    test_queries_processed['tokens'] = test_queries_processed['content'].fillna("").apply(
        lambda txt: [t.lower() for t in word_tokenize(str(txt))]
    )

    # 3. RETRIEVAL MODEL
    # -----------------------------------------------------
    print("--- 3. Running Retrieval Model ---")
    corpus = docs_processed['tokens'].tolist()
    bm25 = BM25Plus(corpus)

    results_train = []
    results_test = []

    # Loop over every train query for evaluation
    for _, row in train_queries_processed.iterrows():
        query_id = str(row["id"])
        query_tokens = row['tokens']
        # BM25 retrieval
        doc_scores = bm25.get_scores(query_tokens)
        topk_indices = np.argsort(doc_scores)[-TOP_K:][::-1]
        top_docs = docs_processed["id"].iloc[topk_indices].tolist()
        # Make sure returned doc ids are strings
        top_docs = [str(d) for d in top_docs]
        result = {"query_id": query_id, "relevant_docs": top_docs}
        results_train.append(result)

    # Loop over every test query for submission
    for _, row in test_queries_processed.iterrows():
        query_id = str(row["id"])
        query_tokens = row['tokens']
        doc_scores = bm25.get_scores(query_tokens)
        topk_indices = np.argsort(doc_scores)[-TOP_K:][::-1]
        top_docs = docs_processed["id"].iloc[topk_indices].tolist()
        top_docs = [str(d) for d in top_docs]
        result = {"query_id": query_id, "relevant_docs": top_docs}
        results_test.append(result)

    # Optional: quick debug print for first 3 queries
    print("Sample retrieval (first 3 train queries):")
    for sample in results_train[:3]:
        print(sample)

    # Evaluate the retrieval results
    print("--- Evaluating Retrieval ---")
    metrics = evaluate_retrieval(results_train, qrels_processed, TOP_K)

    print(f"Average Recall@{TOP_K}: {metrics['avg_recall']:.4f}")
    print(f"Average Precision@{TOP_K}: {metrics['avg_precision']:.4f}")
    print(f"MRR@{TOP_K}: {metrics['mrr']:.4f}")

    # 4. FORMAT SUBMISSION
    # -----------------------------------------------------
    print("--- 4. Saving Submission ---")
    results_df = pd.DataFrame(results_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "submission_baseline.csv")
    results_df.to_csv(output_path, index=False)

    print(f"Done! submission_baseline was saved to {output_path}")


if __name__ == "__main__":
    run_baseline()