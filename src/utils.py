import json
import pandas as pd
import os


def load_data(data_path):

    # 1. file paths
    # os.path.join so it works on windows, mac...
    docs_path = os.path.join(data_path, "docs.json")
    train_queries_path = os.path.join(data_path, "queries_train.json")
    test_queries_path = os.path.join(data_path, "queries_test.json")
    qgts_path = os.path.join(data_path, "qgts_train.json")

    # 2. load docs
    dataframe_docs = pd.read_json(docs_path)

    # 3. load Queries (Train & Test)
    print(f"---- loading {train_queries_path}...")
    train_queries_df = pd.read_json(train_queries_path)
    print(f"---- loading {test_queries_path}...")
    test_queries_df = pd.read_json(test_queries_path)

    # 4. load queries ground truths (qgts)
    print(f"---- loading {qgts_path}...")
    with open(qgts_path, "r") as f:
        query_ground_truth_data = json.load(f)

    print("Data loading complete.")

    return dataframe_docs, train_queries_df, test_queries_df, query_ground_truth_data
