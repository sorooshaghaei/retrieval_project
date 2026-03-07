"""Retrieval models: TF-IDF, BM25, and dense retrieval."""

from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Simple tokenizer used by BM25."""
    # Match notebook and pipeline behavior: lowercase + basic symbol splitting.
    normalized = str(text or "").lower()
    normalized = re.sub(r"[-_/]", " ", normalized)
    return _TOKEN_PATTERN.findall(normalized)


def _build_result(query_id: str, ranked_doc_ids: List[str]) -> dict:
    return {"query_id": str(query_id), "relevant_docs": [str(doc_id) for doc_id in ranked_doc_ids]}


def _load_sentence_transformer(model_name: str):
    """Load a sentence-transformers model lazily so lexical pipelines stay lightweight."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def run_tfidf_search(docs_df: pd.DataFrame, query_df: pd.DataFrame, top_k: int = 10) -> List[dict]:
    """Retrieve documents with cosine similarity on TF-IDF vectors."""
    top_k = min(top_k, len(docs_df))

    # Use a slightly richer lexical representation (unigrams + bigrams).
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
    try:
        doc_vectors = vectorizer.fit_transform(docs_df["content"])
    except ValueError as error:
        # Small datasets may lose all terms with min_df=2. Fall back to min_df=1.
        if "After pruning, no terms remain" not in str(error):
            raise
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        doc_vectors = vectorizer.fit_transform(docs_df["content"])

    query_vectors = vectorizer.transform(query_df["content"])

    # Similarity matrix shape: [num_queries, num_docs].
    similarity_matrix = cosine_similarity(query_vectors, doc_vectors)
    doc_ids = docs_df["id"].astype(str).to_numpy()

    results: List[dict] = []
    for row_idx, scores in enumerate(similarity_matrix):
        # Highest scores are the most relevant docs for this query.
        top_indices = np.argsort(scores)[-top_k:][::-1]
        ranked_doc_ids = doc_ids[top_indices].tolist()
        results.append(_build_result(query_df.iloc[row_idx]["id"], ranked_doc_ids))

    return results


def run_bm25_search(docs_df: pd.DataFrame, query_df: pd.DataFrame, top_k: int = 10) -> List[dict]:
    """Retrieve documents with BM25+ lexical ranking."""
    top_k = min(top_k, len(docs_df))

    # BM25 works on tokenized corpus rather than raw strings.
    tokenized_corpus = [tokenize(text) for text in docs_df["content"]]
    bm25 = BM25Plus(tokenized_corpus)

    doc_ids = docs_df["id"].astype(str).to_numpy()
    results: List[dict] = []

    for _, row in query_df.iterrows():
        query_tokens = tokenize(row["content"])
        scores = bm25.get_scores(query_tokens)
        # Keep only top-k indices to avoid sorting all docs fully in final output.
        top_indices = np.argsort(scores)[-top_k:][::-1]
        ranked_doc_ids = doc_ids[top_indices].tolist()
        results.append(_build_result(row["id"], ranked_doc_ids))

    return results


def run_dense_search(
    docs_df: pd.DataFrame,
    query_df: pd.DataFrame,
    top_k: int = 10,
    batch_size: int = 128,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[dict]:
    """Retrieve documents with sentence-transformer embeddings.

    Note: this is slower and memory-intensive on large corpora.
    """
    top_k = min(top_k, len(docs_df))

    # Bi-encoder maps texts to dense vectors in a shared semantic space.
    model = _load_sentence_transformer(model_name)
    doc_embeddings = model.encode(
        docs_df["content"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    query_embeddings = model.encode(
        query_df["content"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # With normalized embeddings, dot product == cosine similarity.
    scores = query_embeddings @ doc_embeddings.T
    # Argpartition gets top-k candidates faster than full sort on each row.
    topk_idx = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]

    # Sort only the top-k candidates by score for stable ranked output.
    sorted_topk = topk_idx[
        np.arange(scores.shape[0])[:, None],
        np.argsort(-scores[np.arange(scores.shape[0])[:, None], topk_idx]),
    ]

    doc_ids = docs_df["id"].astype(str).to_numpy()
    results: List[dict] = []

    for query_idx, doc_indices in enumerate(sorted_topk):
        ranked_doc_ids = doc_ids[doc_indices].tolist()
        results.append(_build_result(query_df.iloc[query_idx]["id"], ranked_doc_ids))

    return results
