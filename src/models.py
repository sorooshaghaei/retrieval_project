# TF-IDF, BM25, embeded_retrieve logic goes here

import pandas as pd

# BM25 and TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Plus

# SentenceTransformer (Deep Learning)
from sentence_transformers import SentenceTransformer
import numpy as np

import re

# tokenizing for bm25
_token_re = re.compile(r"[a-z0-9]+")  

def tokenize(text: str):
    if text is None:
        return []
    text = str(text).lower()
    # "bla-bla" match "bla bla"
    text = re.sub(r"[-_/]", " ", text)
    return _token_re.findall(text)

def embed_retrieve(docs_df, queries_df, top_k=10, batch_size=128, model_name="all-MiniLM-L6-v2"):
    # Step 1: Load a sentence-embedding model (bi-encoder).
    # This model maps texts into a shared vector space.
    print(f"loadiing '{model_name}' (this might take a minute!)...")
    model = SentenceTransformer(model_name)

    # Step 2: Encode documents into dense vectors -> This turns text into a matrix of numbers
    # normalize_embeddings=True makes dot product equal cosine similarity.
    print(f"encoding {len(docs_df)} documents...")
    doc_emb = model.encode(
        docs_df["content"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Step 3: Encode queries into dense vectors in the same space.
    print("encoding queries...")
    qry_emb = model.encode(
        queries_df["content"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Step 4: Compute similarity between every query and every document.
    print("calculating Similarity...")
    # With normalized vectors, dot product == cosine similarity
    scores = qry_emb @ doc_emb.T

    # Step 5: Select top_k documents per query.
    # argpartition is faster than full sort for large matrices.
    top_k = min(top_k, len(docs_df))  # safety

    # argpartition kth is 0-indexed -> use top_k - 1
    topk_idx = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]


    # Step 6: Sort those top_k docs by score (descending).
    topk_sorted = topk_idx[
        np.arange(scores.shape[0])[:, None],
        np.argsort(-scores[np.arange(scores.shape[0])[:, None], topk_idx])
    ]
    # Step 7: Convert indices -> doc IDs and output dicts
    results = []
    for i, doc_indices in enumerate(topk_sorted):
        query_id = queries_df.iloc[i]["id"]
        relevant_docs = docs_df.iloc[doc_indices]["id"].tolist()
        results.append({"query_id": query_id, "relevant_docs": relevant_docs})

    return results


def run_tfidf_search(docs_df, query_df, top_k=10):

    print(f"training tf-idf on {len(docs_df)} docs.....")

    # it needs raw strings
    # use the 'content' column created in preprocessing
    vectorizer = TfidfVectorizer(lowercase=True)
    doc_vectors = vectorizer.fit_transform(docs_df["content"])
    query_vectors = vectorizer.transform(query_df["content"])

    print("now calculating Cosine Similarity..")
    similarity_matrix = cosine_similarity(query_vectors, doc_vectors)

    results = []
    # Convert matrix to list of results
    for i, row in enumerate(similarity_matrix):
        query_id = query_df.iloc[i]["id"]
        top_indices = row.argsort()[-top_k:][::-1]
        relevant_docs = docs_df.iloc[top_indices]["id"].tolist()
        results.append({"query_id": query_id, "relevant_docs": relevant_docs})

    return results


def run_bm25_search(docs_df, query_df, top_k=10):
    print(f"indexing bm25 on {len(docs_df)} docs...")

    tokenized_corpus = [tokenize(doc) for doc in docs_df["content"]]
    bm25 = BM25Plus(tokenized_corpus)

    results = []
    print("Retrieving...")

    for _, row in query_df.iterrows():
        query_id = row["id"]
        tokenized_query = tokenize(row["content"])

        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        relevant_docs = docs_df.iloc[top_indices]["id"].tolist()
        results.append({"query_id": query_id, "relevant_docs": relevant_docs})

    return results