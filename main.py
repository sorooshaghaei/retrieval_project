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
from src.preprocess import create_content_column


DATA_DIR = "data"  # Folder where json files are
OUTPUT_DIR = "outputs"  # Folder where submission.csv goes
TOP_K = 10  # How many docs to retrieve per query


def run_baseline():
    # 1. LOAD DATA
    # -----------------------------------------------------
    print("--- 1. Loading Data ---")
    docs_df, train_queries_df, test_queries_df, qrels = load_data(DATA_DIR)

    # 2. PREPROCESS
    # -----------------------------------------------------
    print("--- 2. Preprocessing ---")
    # For Documents: Merge title + text + tags
    print("Processing Documents...")
    docs_processed = create_content_column(docs_df, ["title", "text", "tags"])

    # For Queries: Merge title + text (queries might not have tags, check first)
    print("Processing Queries...")
    # Note: Adjust columns based on what exists in queries_test.json
    test_queries_processed = create_content_column(test_queries_df, ["title", "text"])

    # 3. RETRIEVAL MODEL
    # -----------------------------------------------------
    print("--- 3. Running Retrieval Model ---")

    # Currently, this is a DUMMY implementation that just returns random docs.
    # TEAMATES TODO: it will be replaced with TF-IDF or BM25 later.

    results = []

    # Loop over every test query
    for index, row in test_queries_processed.iterrows():
        query_id = row["id"]

        # ------------ START MODEL LOGIC ------------
        # (This is where we will add TF-IDF later)
        # foe now, let's just take the first 10 docs as a placeholder
        top_docs = docs_processed["id"].head(TOP_K).tolist()
        # ------------ END MODEL LOGIC ------------

        results.append({"query_id": query_id, "relevant_docs": top_docs})

    # 4. FORMAT SUBMISSION
    # -----------------------------------------------------
    print("--- 4. Saving Submission ---")
    # Kaggle requires specific format
    # This part depends on the specific Kaggle submission.csv format
    # will be refined this later....

    results_df = pd.DataFrame(results)

    # make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "submission_baseline.csv")
    results_df.to_csv(output_path, index=False)

    print(f"Done! submission_baseline was saved to {output_path}")


if __name__ == "__main__":
    run_baseline()
