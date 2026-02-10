# TF-IDF, BM25 logic goes here

from sentence_transformers import SentenceTransformer
import numpy as np

def embed_retrieve(docs_df, queries_df, top_k=10, batch_size=128, model_name="all-MiniLM-L6-v2"):
    # Step 1: Load a sentence-embedding model (bi-encoder).
    # This model maps texts into a shared vector space.
    model = SentenceTransformer(model_name)

    # Step 2: Encode documents into dense vectors.
    # normalize_embeddings=True makes dot product equal cosine similarity.
    doc_emb = model.encode(
        docs_df["content"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Step 3: Encode queries into dense vectors in the same space.
    qry_emb = model.encode(
        queries_df["content"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Step 4: Compute similarity between every query and every document.
    # With normalized vectors, dot product == cosine similarity.
    scores = qry_emb @ doc_emb.T

    # Step 5: Select top_k documents per query.
    # argpartition is faster than full sort for large matrices.
    topk_idx = np.argpartition(-scores, top_k, axis=1)[:, :top_k]

    # Step 6: Sort those top_k docs by score (descending).
    topk_sorted = topk_idx[
        np.arange(scores.shape[0])[:, None],
        np.argsort(-scores[np.arange(scores.shape[0])[:, None], topk_idx])
    ]

    return topk_sorted