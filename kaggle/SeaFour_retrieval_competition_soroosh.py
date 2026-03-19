"""
# SeaFour — Retrieval Engine Competition Notebook

**Team:** SeaFour  
**Members:** Mehdi (Soroosh) Aghaei, Maksym DOLHOV, Khánh Bảo

This notebook restructures the original submission script into a full experimentation workflow for both phases of the competition:

1. **Phase 1 — Retrieval baselines and optimization**
   - TF-IDF
   - BM25+
   - Embedding-based retrieval / reranking
   - Local validation with Recall, Precision, and MRR

2. **Phase 2 — Classification and retrieval optimization**
   - Query category prediction
   - Category-aware reranking
   - Retrieval/classification cooperation
   - Final submission generation

The notebook is written to be usable in **Kaggle** or locally.  
It also contains a **report section at the end** describing what was improved, why it was improved, and how the final system is organized.
"""

"""
## Why this notebook is different from the original one

The original file is mainly a **submission generator**. It supports TF-IDF, a BM25-style baseline, and an embedding hybrid, then writes the CSV. It does **not** yet provide a clean experimental loop for local validation, it uses `BM25Okapi` instead of **BM25+**, and it does not contain a complete Phase 2 classifier pipeline.  
This version turns the project into an **IR experimentation notebook**: reproducible setup, validation protocol, metric computation, caching, model comparison, reranking, classification, and final reporting.
"""


# Core libraries
from __future__ import annotations

import os
import gc
import re
import csv
import json
import math
import time
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

"""
## Configuration

The configuration below centralizes the main design choices:
- field weighting for title/text/tags
- lexical and dense retrieval settings
- reranking and category bonus settings
- phase switch for submission writing

This makes experiments easier to repeat and compare.
"""


@dataclass
class CFG:
    # Paths
    competition_dir_name: str = "retrieval-engine-competition"
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"

    # Data / preprocessing
    lowercase: bool = True
    use_stopwords: bool = False
    use_stemming: bool = False
    title_weight: int = 3
    text_weight: int = 1
    tags_weight: int = 2
    query_tags_weight: int = 2

    # TF-IDF
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 1
    tfidf_max_df: float = 0.98
    tfidf_sublinear_tf: bool = True

    # BM25+
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_delta: float = 1.0

    # Retrieval depth
    top_k: int = 100
    candidate_k: int = 250

    # Dense retrieval
    embedding_model_names: Tuple[str, ...] = (
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    )
    embedding_batch_size: int = 128
    dense_chunk_size: int = 20000

    # Fusion / reranking
    rrf_k: int = 60
    use_dense_rerank: bool = True
    category_bonus: float = 0.10

    # Validation
    val_size: float = 0.2

    # Submission phase
    phase: int = 2  # 1 or 2

CFG = CFG()
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
Path(CFG.cache_dir).mkdir(parents=True, exist_ok=True)

print(asdict(CFG))

"""
## Data loading

The notebook first tries common Kaggle paths.  
If the full competition data is not present locally, it still loads the uploaded training/query files when available, so classification and metadata analysis can still run.

**Important note:** the uploaded chat files do not include `docs.json`, so full retrieval evaluation requires the Kaggle competition dataset or a local copy of `docs.json`.
"""


def find_data_dir() -> Optional[Path]:
    candidates = [
        Path("/kaggle/input") / CFG.competition_dir_name,
        Path("/kaggle/input/retrieval-engine-competition"),
        Path("."),
        Path("data"),
        Path("../data"),
        Path("/mnt/data"),
    ]
    for base in candidates:
        if (base / "queries_train.json").exists() or (base / "queries_test.json").exists():
            return base
    return None

DATA_DIR = find_data_dir()
print("DATA_DIR:", DATA_DIR)

if DATA_DIR is None:
    raise FileNotFoundError("Could not find any competition data directory.")

def maybe_read_json(path: Path):
    if path.exists():
        return pd.read_json(path)
    return None

queries_train_df = maybe_read_json(DATA_DIR / "queries_train.json")
queries_test_df = maybe_read_json(DATA_DIR / "queries_test.json")
docs_df = maybe_read_json(DATA_DIR / "docs.json")
sample_submission_df = pd.read_csv(DATA_DIR / "submission.csv") if (DATA_DIR / "submission.csv").exists() else None

qgts_path = DATA_DIR / "qgts_train.json"
qgts_train = None
if qgts_path.exists():
    with open(qgts_path, "r", encoding="utf-8") as f:
        qgts_train = json.load(f)

print("queries_train:", None if queries_train_df is None else queries_train_df.shape)
print("queries_test :", None if queries_test_df is None else queries_test_df.shape)
print("docs         :", None if docs_df is None else docs_df.shape)
print("submission   :", None if sample_submission_df is None else sample_submission_df.shape)
print("qgts_train   :", None if qgts_train is None else len(qgts_train))

"""
## Basic sanity checks

These checks are useful before any modeling:
- training category balance
- whether test queries include tags
- whether the qgts file contains relevant document IDs and categories
"""


if queries_train_df is not None:
    display(queries_train_df.head(3))
    if "category" in queries_train_df.columns:
        display(queries_train_df["category"].value_counts().rename("train_category_count").to_frame())

if queries_test_df is not None:
    display(queries_test_df.head(3))

if qgts_train is not None:
    first_key = next(iter(qgts_train))
    print("First qgts key:", first_key)
    print(json.dumps(qgts_train[first_key], indent=2)[:1000])

"""
## Build gold labels from `qgts_train`

For retrieval evaluation we need the gold relevant document IDs per training query.  
For classification evaluation we also keep the gold category.
"""


def build_gold_df(qgts_train: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for qid, item in qgts_train.items():
        rows.append({
            "id": str(qid),
            "gold_doc_ids": [str(x["doc_id"]) for x in item.get("relevant_doc_ids", [])],
            "gold_category": item.get("category", None),
            "total_relevant_docs": int(item.get("total_relevant_docs", 0)),
        })
    return pd.DataFrame(rows)

gold_df = build_gold_df(qgts_train) if qgts_train is not None else None
if gold_df is not None:
    display(gold_df.head())

"""
## Merge training queries with gold labels

This unified table will be used for:
- train/validation splitting
- classifier training
- retrieval metric computation
"""


train_df = None
if queries_train_df is not None and gold_df is not None:
    train_df = queries_train_df.merge(gold_df, on="id", how="inner")
    display(train_df.head())
    print("Merged train rows:", len(train_df))

"""
## Preprocessing

The original notebook already had lowercasing, optional stopword removal, optional stemming, and a shared `content` field.  
This notebook keeps that idea but makes it explicit and configurable.

Two design principles matter here:

1. **Field weighting**  
   Title and tags often carry denser signal than long free text in IR tasks.

2. **Stable preprocessing**  
   The same pipeline should be used across TF-IDF, BM25+, classifier features, and dense input construction where relevant.
"""


token_pattern = re.compile(r"[a-z0-9]+")

_nltk_ready = False
_nltk_stopwords = None
_nltk_stemmer = None

def init_nltk():
    global _nltk_ready, _nltk_stopwords, _nltk_stemmer
    if _nltk_ready:
        return True
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        _nltk_stopwords = set(stopwords.words("english"))
        _nltk_stemmer = PorterStemmer()
        _nltk_ready = True
        return True
    except Exception:
        _nltk_ready = False
        return False

def tokenize(text: str) -> List[str]:
    txt = "" if text is None else str(text)
    if CFG.lowercase:
        txt = txt.lower()
    txt = re.sub(r"[-_/]", " ", txt)
    tokens = token_pattern.findall(txt)

    if not (CFG.use_stopwords or CFG.use_stemming):
        return tokens

    if not init_nltk():
        return tokens

    if CFG.use_stopwords:
        tokens = [t for t in tokens if t not in _nltk_stopwords]
    if CFG.use_stemming:
        tokens = [_nltk_stemmer.stem(t) for t in tokens]

    return tokens

def value_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(map(str, value))
    if pd.isna(value):
        return ""
    return str(value)

def weighted_join(row, spec: Dict[str, int]) -> str:
    parts = []
    for col, weight in spec.items():
        text = value_to_text(row[col]) if col in row else ""
        for _ in range(max(0, int(weight))):
            parts.append(text)
    joined = " ".join(parts).strip()
    return joined.lower() if CFG.lowercase else joined

def add_content_columns(df: pd.DataFrame, is_query: bool = False) -> pd.DataFrame:
    out = df.copy()
    for col in ["title", "text", "tags"]:
        if col not in out.columns:
            out[col] = ""
    if is_query:
        spec = {
            "title": CFG.title_weight,
            "text": CFG.text_weight,
            "tags": CFG.query_tags_weight,
        }
    else:
        spec = {
            "title": CFG.title_weight,
            "text": CFG.text_weight,
            "tags": CFG.tags_weight,
        }
    out["content"] = out.apply(lambda row: weighted_join(row, spec), axis=1)
    out["content_no_tags"] = out.apply(
        lambda row: weighted_join(row, {"title": CFG.title_weight, "text": CFG.text_weight}),
        axis=1,
    )
    out["id"] = out["id"].astype(str)
    return out

if queries_train_df is not None:
    queries_train_df = add_content_columns(queries_train_df, is_query=True)
if queries_test_df is not None:
    queries_test_df = add_content_columns(queries_test_df, is_query=True)
if docs_df is not None:
    docs_df = add_content_columns(docs_df, is_query=False)
if train_df is not None:
    train_df = add_content_columns(train_df, is_query=True)

print("Prepared content columns.")

"""
## Tags are highly informative for Phase 2

Before training a classifier, it is worth checking how informative the query tags already are.  
In this dataset, tags are not noise. They are a strong structured signal. If a query tag consistently maps to one category, using that information is legitimate and useful because the tags are part of the provided input.

This leads to an effective strategy:
- **first** try a deterministic tag-to-category mapping
- **then** fall back to a text classifier only when tags are missing or ambiguous
"""


def build_tag_category_map(train_df: pd.DataFrame) -> Dict[str, Counter]:
    tag_to_cat = defaultdict(Counter)
    for _, row in train_df.iterrows():
        for tag in row.get("tags", []) or []:
            tag_to_cat[str(tag)][str(row["gold_category"] if "gold_category" in row else row["category"])] += 1
    return tag_to_cat

tag_to_cat = build_tag_category_map(train_df) if train_df is not None else {}
tag_summary = []
for tag, counts in tag_to_cat.items():
    total = sum(counts.values())
    major_cat, major_count = counts.most_common(1)[0]
    tag_summary.append({
        "tag": tag,
        "majority_category": major_cat,
        "purity": major_count / total,
        "count": total,
        "distribution": dict(counts),
    })
tag_summary_df = pd.DataFrame(tag_summary).sort_values(["purity", "count"], ascending=[False, False])
display(tag_summary_df)

"""
## Train / validation split

A stratified split keeps the category proportions stable.  
This is important because the competition score later includes classification accuracy, and category imbalance can distort model comparison.
"""


if train_df is not None:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=CFG.val_size, random_state=SEED)
    y_for_split = train_df["gold_category"]
    train_idx, val_idx = next(splitter.split(train_df, y_for_split))

    train_fold = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold = train_df.iloc[val_idx].reset_index(drop=True)

    print("train_fold:", train_fold.shape)
    print("val_fold  :", val_fold.shape)

    display(train_fold["gold_category"].value_counts().rename("train_fold_count").to_frame())
    display(val_fold["gold_category"].value_counts().rename("val_fold_count").to_frame())

"""
## Metrics

The competition uses:
- Recall
- Precision
- MRR
- Accuracy (Phase 2)

We implement them locally to compare systems before generating a submission.
"""


def recall_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    pred = pred_ids[:k]
    gold = set(gold_ids)
    if len(gold) == 0:
        return 0.0
    hit = sum(1 for x in pred if x in gold)
    return hit / len(gold)

def precision_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    pred = pred_ids[:k]
    if len(pred) == 0:
        return 0.0
    gold = set(gold_ids)
    hit = sum(1 for x in pred if x in gold)
    return hit / len(pred)

def mrr_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    gold = set(gold_ids)
    for rank, doc_id in enumerate(pred_ids[:k], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(pred_map: Dict[str, List[str]], gold_table: pd.DataFrame, k: int) -> Dict[str, float]:
    recalls, precisions, mrrs = [], [], []
    for _, row in gold_table.iterrows():
        qid = str(row["id"])
        pred = [str(x) for x in pred_map.get(qid, [])]
        gold = [str(x) for x in row["gold_doc_ids"]]
        recalls.append(recall_at_k(pred, gold, k))
        precisions.append(precision_at_k(pred, gold, k))
        mrrs.append(mrr_at_k(pred, gold, k))
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"precision@{k}": float(np.mean(precisions)),
        f"mrr@{k}": float(np.mean(mrrs)),
    }

def evaluate_classification(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred))}

def combined_score(metrics: Dict[str, float], k: int, phase: int = 2) -> float:
    recall = metrics.get(f"recall@{k}", 0.0)
    precision = metrics.get(f"precision@{k}", 0.0)
    mrr = metrics.get(f"mrr@{k}", 0.0)
    acc = metrics.get("accuracy", 0.0) if phase == 2 else 0.0
    return 0.25 * recall + 0.25 * precision + 0.25 * mrr + 0.25 * acc

"""
## Phase 2 classifier

We use a **two-stage classifier**:

1. **Tag-based rule**  
   If the query contains tags already seen in training, predict the majority category implied by those tags.

2. **Text classifier fallback**  
   If tags are missing or inconclusive, use a TF-IDF + Logistic Regression classifier over the query text.

This design is deliberately simple, strong, interpretable, and fast.
"""


class QueryCategoryClassifier:
    def __init__(self, train_df: pd.DataFrame):
        self.tag_to_cat = build_tag_category_map(train_df)
        self.text_model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=False,
                tokenizer=tokenize,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED,
            )),
        ])
        self.is_fit = False

    def predict_from_tags(self, tags: Iterable[str]) -> Optional[str]:
        counts = Counter()
        for tag in tags or []:
            tag = str(tag)
            if tag in self.tag_to_cat:
                counts.update(self.tag_to_cat[tag])
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def fit(self, train_df: pd.DataFrame):
        self.text_model.fit(train_df["content_no_tags"], train_df["gold_category"])
        self.is_fit = True
        return self

    def predict(self, df: pd.DataFrame) -> List[str]:
        assert self.is_fit, "Call fit() first."
        text_preds = self.text_model.predict(df["content_no_tags"])
        out = []
        for i, (_, row) in enumerate(df.iterrows()):
            tag_pred = self.predict_from_tags(row.get("tags", []))
            out.append(tag_pred if tag_pred is not None else text_preds[i])
        return out

if train_fold is not None:
    clf_model = QueryCategoryClassifier(train_fold).fit(train_fold)
    val_cat_pred = clf_model.predict(val_fold)
    clf_metrics = evaluate_classification(val_fold["gold_category"].tolist(), val_cat_pred)
    print(clf_metrics)
    print(classification_report(val_fold["gold_category"], val_cat_pred))

"""
## Optional document pseudo-category inference

To let classification help retrieval, we need a way to estimate document categories.  
If document tags are available, we infer a pseudo-category using the same tag-to-category map learned from the training queries.

This is not guaranteed to be perfect, but it is often strong enough to support a **category bonus** during reranking.
"""


def infer_category_from_tags(tags: Iterable[str], tag_to_cat: Dict[str, Counter]) -> Tuple[Optional[str], float]:
    counts = Counter()
    for tag in tags or []:
        tag = str(tag)
        if tag in tag_to_cat:
            counts.update(tag_to_cat[tag])
    if not counts:
        return None, 0.0
    total = sum(counts.values())
    cat, count = counts.most_common(1)[0]
    return cat, count / total

if docs_df is not None:
    inferred = docs_df["tags"].apply(lambda x: infer_category_from_tags(x, tag_to_cat))
    docs_df["pseudo_category"] = inferred.apply(lambda x: x[0])
    docs_df["pseudo_category_conf"] = inferred.apply(lambda x: x[1])
    display(docs_df[["id", "tags", "pseudo_category", "pseudo_category_conf"]].head())
else:
    print("docs.json is not available here, so document pseudo-category inference will run once docs are present.")

"""
## Retrieval models

We implement three families of models:

1. **TF-IDF**
2. **BM25+**
3. **Embedding-based retrieval**

Then we add:
- **RRF fusion** across models
- **dense reranking**
- **category-aware score bonus**
"""


def require_docs():
    if docs_df is None:
        raise FileNotFoundError(
            "docs.json is required for retrieval experiments. "
            "The uploaded chat files do not include docs.json, so run this cell in Kaggle or add docs.json locally."
        )


# BM25+ import
try:
    from rank_bm25 import BM25Plus
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rank_bm25"])
    from rank_bm25 import BM25Plus

"""
### TF-IDF retriever
"""


class TfidfRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            ngram_range=CFG.tfidf_ngram_range,
            min_df=CFG.tfidf_min_df,
            max_df=CFG.tfidf_max_df,
            sublinear_tf=CFG.tfidf_sublinear_tf,
        )
        self.doc_matrix = None
        self.doc_ids = None

    def fit(self, docs: pd.DataFrame):
        self.doc_ids = docs["id"].astype(str).tolist()
        self.doc_matrix = self.vectorizer.fit_transform(docs["content"])
        return self

    def search(self, queries: pd.DataFrame, top_k: int) -> Dict[str, List[str]]:
        qmat = self.vectorizer.transform(queries["content"])
        scores = qmat @ self.doc_matrix.T
        pred_map = {}
        for i, qid in enumerate(queries["id"].astype(str).tolist()):
            row = scores.getrow(i).toarray().ravel()
            idx = np.argpartition(row, -top_k)[-top_k:]
            idx = idx[np.argsort(row[idx])[::-1]]
            pred_map[qid] = [self.doc_ids[j] for j in idx]
        return pred_map

"""
### BM25+ retriever
"""


class BM25PlusRetriever:
    def __init__(self):
        self.model = None
        self.doc_ids = None
        self.tokenized_docs = None

    def fit(self, docs: pd.DataFrame):
        self.doc_ids = docs["id"].astype(str).tolist()
        self.tokenized_docs = [tokenize(x) for x in docs["content"].tolist()]
        self.model = BM25Plus(
            self.tokenized_docs,
            k1=CFG.bm25_k1,
            b=CFG.bm25_b,
            delta=CFG.bm25_delta,
        )
        return self

    def search(self, queries: pd.DataFrame, top_k: int) -> Dict[str, List[str]]:
        pred_map = {}
        for _, row in queries.iterrows():
            qid = str(row["id"])
            tokens = tokenize(row["content"])
            scores = np.array(self.model.get_scores(tokens))
            idx = np.argpartition(scores, -top_k)[-top_k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
            pred_map[qid] = [self.doc_ids[j] for j in idx]
        return pred_map

"""
### Dense retriever

This cell supports embedding models from `sentence-transformers`.  
Embeddings are cached to disk to avoid recomputing the document matrix every time.

To keep memory usage under control, retrieval is done in **chunks** across the document embedding matrix.
"""


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer


def embedding_cache_path(model_name: str, n_rows: int, prefix: str) -> Path:
    safe = model_name.replace("/", "__")
    return Path(CFG.cache_dir) / f"{prefix}_{safe}_{n_rows}.npy"

def encode_texts(model_name: str, texts: List[str], prefix: str, batch_size: int) -> np.ndarray:
    cache_path = embedding_cache_path(model_name, len(texts), prefix)
    if cache_path.exists():
        arr = np.load(cache_path)
        return arr.astype(np.float32)

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(cache_path, emb)
    return emb

def dense_topk_search(query_emb: np.ndarray, doc_emb: np.ndarray, top_k: int, chunk_size: int) -> np.ndarray:
    n_queries = query_emb.shape[0]
    top_scores = np.full((n_queries, top_k), -1e9, dtype=np.float32)
    top_indices = np.full((n_queries, top_k), -1, dtype=np.int64)

    for start in range(0, len(doc_emb), chunk_size):
        stop = min(start + chunk_size, len(doc_emb))
        block = doc_emb[start:stop]
        scores = query_emb @ block.T  # cosine similarity because vectors are normalized

        local_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        local_scores = np.take_along_axis(scores, local_idx, axis=1)
        local_idx = local_idx + start

        merged_scores = np.concatenate([top_scores, local_scores], axis=1)
        merged_indices = np.concatenate([top_indices, local_idx], axis=1)

        keep = np.argpartition(merged_scores, -top_k, axis=1)[:, -top_k:]
        top_scores = np.take_along_axis(merged_scores, keep, axis=1)
        top_indices = np.take_along_axis(merged_indices, keep, axis=1)

        order = np.argsort(top_scores, axis=1)[:, ::-1]
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        top_indices = np.take_along_axis(top_indices, order, axis=1)

    return top_indices

class DenseRetriever:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.doc_ids = None
        self.doc_embeddings = None

    def fit(self, docs: pd.DataFrame):
        self.doc_ids = docs["id"].astype(str).tolist()
        self.doc_embeddings = encode_texts(
            self.model_name,
            docs["content"].tolist(),
            prefix="doc_emb",
            batch_size=CFG.embedding_batch_size,
        )
        return self

    def search(self, queries: pd.DataFrame, top_k: int) -> Dict[str, List[str]]:
        q_emb = encode_texts(
            self.model_name,
            queries["content"].tolist(),
            prefix="query_emb_tmp",
            batch_size=CFG.embedding_batch_size,
        )
        idx = dense_topk_search(q_emb, self.doc_embeddings, top_k, CFG.dense_chunk_size)
        pred_map = {}
        query_ids = queries["id"].astype(str).tolist()
        for i, qid in enumerate(query_ids):
            pred_map[qid] = [self.doc_ids[j] for j in idx[i]]
        return pred_map

"""
## Fusion and reranking

We use **Reciprocal Rank Fusion (RRF)** as a robust combination strategy across lexical and dense rankings.

Then, if a predicted query category and a document pseudo-category match, a small **category bonus** is added.  
This lets the classifier support retrieval without hard-filtering the candidate set too aggressively.
"""


def rrf_fuse(rankings: List[Dict[str, List[str]]], top_k: int, rrf_k: int = 60) -> Dict[str, List[str]]:
    all_qids = rankings[0].keys()
    fused = {}
    for qid in all_qids:
        scores = defaultdict(float)
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking[qid], start=1):
                scores[doc_id] += 1.0 / (rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        fused[qid] = [doc_id for doc_id, _ in ranked]
    return fused

def add_category_bonus(
    pred_map: Dict[str, List[str]],
    query_category_map: Dict[str, str],
    docs_df: pd.DataFrame,
    bonus: float,
    top_k: int,
) -> Dict[str, List[str]]:
    if "pseudo_category" not in docs_df.columns:
        return pred_map

    doc_cat = docs_df.set_index("id")["pseudo_category"].to_dict()
    doc_conf = docs_df.set_index("id")["pseudo_category_conf"].to_dict()

    boosted = {}
    for qid, docs in pred_map.items():
        qcat = query_category_map.get(qid, None)
        scored = []
        for rank, doc_id in enumerate(docs, start=1):
            score = 1.0 / (CFG.rrf_k + rank)
            if qcat is not None and doc_cat.get(doc_id) == qcat:
                score += bonus * float(doc_conf.get(doc_id, 1.0))
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        boosted[qid] = [doc_id for doc_id, _ in scored[:top_k]]
    return boosted

"""
## Experiment runner

This section evaluates several systems on the validation split and stores the results in a comparison table.

Suggested sequence:
1. TF-IDF baseline
2. BM25+ baseline
3. Dense baseline(s)
4. TF-IDF + BM25+ fusion
5. TF-IDF + BM25+ + dense fusion
6. Category-aware reranking
"""


def run_validation_experiments(train_fold: pd.DataFrame, val_fold: pd.DataFrame, docs_df: pd.DataFrame) -> pd.DataFrame:
    require_docs()

    # Train classifier for category predictions used in Phase 2
    clf_model = QueryCategoryClassifier(train_fold).fit(train_fold)
    val_cat_pred = clf_model.predict(val_fold)
    query_category_map = dict(zip(val_fold["id"].astype(str), val_cat_pred))

    experiments = []

    # 1) TF-IDF
    tfidf = TfidfRetriever().fit(docs_df)
    tfidf_pred = tfidf.search(val_fold, CFG.top_k)
    tfidf_metrics = evaluate_retrieval(tfidf_pred, val_fold, CFG.top_k)
    tfidf_metrics.update(evaluate_classification(val_fold["gold_category"], val_cat_pred))
    tfidf_metrics["score"] = combined_score(tfidf_metrics, CFG.top_k, phase=CFG.phase)
    tfidf_metrics["model"] = "tfidf"
    experiments.append(tfidf_metrics)

    # 2) BM25+
    bm25p = BM25PlusRetriever().fit(docs_df)
    bm25_pred = bm25p.search(val_fold, CFG.top_k)
    bm25_metrics = evaluate_retrieval(bm25_pred, val_fold, CFG.top_k)
    bm25_metrics.update(evaluate_classification(val_fold["gold_category"], val_cat_pred))
    bm25_metrics["score"] = combined_score(bm25_metrics, CFG.top_k, phase=CFG.phase)
    bm25_metrics["model"] = "bm25plus"
    experiments.append(bm25_metrics)

    # 3) Lexical fusion
    lex_fused = rrf_fuse([tfidf_pred, bm25_pred], top_k=CFG.top_k, rrf_k=CFG.rrf_k)
    lex_metrics = evaluate_retrieval(lex_fused, val_fold, CFG.top_k)
    lex_metrics.update(evaluate_classification(val_fold["gold_category"], val_cat_pred))
    lex_metrics["score"] = combined_score(lex_metrics, CFG.top_k, phase=CFG.phase)
    lex_metrics["model"] = "tfidf+bm25plus_rrf"
    experiments.append(lex_metrics)

    # 4) Dense models and dense fusion
    for model_name in CFG.embedding_model_names:
        dense = DenseRetriever(model_name).fit(docs_df)
        dense_pred = dense.search(val_fold, CFG.top_k)

        dense_metrics = evaluate_retrieval(dense_pred, val_fold, CFG.top_k)
        dense_metrics.update(evaluate_classification(val_fold["gold_category"], val_cat_pred))
        dense_metrics["score"] = combined_score(dense_metrics, CFG.top_k, phase=CFG.phase)
        dense_metrics["model"] = f"dense::{model_name}"
        experiments.append(dense_metrics)

        fused = rrf_fuse([tfidf_pred, bm25_pred, dense_pred], top_k=CFG.top_k, rrf_k=CFG.rrf_k)
        fused_metrics = evaluate_retrieval(fused, val_fold, CFG.top_k)
        fused_metrics.update(evaluate_classification(val_fold["gold_category"], val_cat_pred))
        fused_metrics["score"] = combined_score(fused_metrics, CFG.top_k, phase=CFG.phase)
        fused_metrics["model"] = f"rrf::{model_name}"
        experiments.append(fused_metrics)

        if "pseudo_category" in docs_df.columns:
            boosted = add_category_bonus(fused, query_category_map, docs_df, CFG.category_bonus, CFG.top_k)
            boosted_metrics = evaluate_retrieval(boosted, val_fold, CFG.top_k)
            boosted_metrics.update(evaluate_classification(val_fold["gold_category"], val_cat_pred))
            boosted_metrics["score"] = combined_score(boosted_metrics, CFG.top_k, phase=CFG.phase)
            boosted_metrics["model"] = f"rrf+category_bonus::{model_name}"
            experiments.append(boosted_metrics)

        gc.collect()

    result_df = pd.DataFrame(experiments).sort_values("score", ascending=False).reset_index(drop=True)
    return result_df

if docs_df is not None and train_fold is not None:
    validation_results = run_validation_experiments(train_fold, val_fold, docs_df)
    display(validation_results)
else:
    print("Validation experiments are ready, but docs.json is required to execute retrieval.")

"""
## Final training on all training queries

Once the best configuration is identified on the validation split, retrain on all labeled queries and generate predictions for the test set.

In Phase 1:
- category column must be empty (`""`)

In Phase 2:
- category column must contain the predicted label
"""


def choose_best_dense_model(validation_results: pd.DataFrame) -> Optional[str]:
    if validation_results is None or len(validation_results) == 0:
        return None
    dense_rows = validation_results[validation_results["model"].str.startswith("rrf+category_bonus::")]
    if len(dense_rows) == 0:
        dense_rows = validation_results[validation_results["model"].str.startswith("rrf::")]
    if len(dense_rows) == 0:
        return None
    best_model = dense_rows.iloc[0]["model"].split("::", 1)[1]
    return best_model

best_dense_model = None
if "validation_results" in globals():
    best_dense_model = choose_best_dense_model(validation_results)
print("best_dense_model:", best_dense_model)


def train_final_system_and_predict(
    train_df: pd.DataFrame,
    queries_test_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    dense_model_name: Optional[str] = None,
):
    require_docs()

    # Classification model
    clf_model = QueryCategoryClassifier(train_df).fit(train_df)
    test_cat_pred = clf_model.predict(queries_test_df)
    test_query_category_map = dict(zip(queries_test_df["id"].astype(str), test_cat_pred))

    # Retrieval models
    tfidf = TfidfRetriever().fit(docs_df)
    bm25p = BM25PlusRetriever().fit(docs_df)
    tfidf_pred = tfidf.search(queries_test_df, CFG.top_k)
    bm25_pred = bm25p.search(queries_test_df, CFG.top_k)

    rankings = [tfidf_pred, bm25_pred]

    if dense_model_name is not None:
        dense = DenseRetriever(dense_model_name).fit(docs_df)
        dense_pred = dense.search(queries_test_df, CFG.top_k)
        rankings.append(dense_pred)

    final_pred = rrf_fuse(rankings, top_k=CFG.top_k, rrf_k=CFG.rrf_k)

    if "pseudo_category" in docs_df.columns and CFG.phase == 2:
        final_pred = add_category_bonus(
            final_pred,
            test_query_category_map,
            docs_df,
            bonus=CFG.category_bonus,
            top_k=CFG.top_k,
        )

    return final_pred, test_cat_pred

if docs_df is not None and train_df is not None and queries_test_df is not None:
    final_pred_map, final_test_categories = train_final_system_and_predict(
        train_df=train_df,
        queries_test_df=queries_test_df,
        docs_df=docs_df,
        dense_model_name=best_dense_model if best_dense_model is not None else CFG.embedding_model_names[0],
    )
    print("Final predictions generated.")
else:
    print("Final prediction cell is ready, but docs.json is required.")

"""
## Submission writer

The writer below matches the expected three-column format:
- `query_id`
- `relevant_doc_ids`
- `category`

For Phase 1, category is written as an empty string.  
For Phase 2, category is filled with the classifier prediction.
"""


def write_submission(
    queries_test_df: pd.DataFrame,
    pred_map: Dict[str, List[str]],
    category_preds: Optional[List[str]],
    output_path: Path,
    phase: int,
):
    rows = []
    qids = queries_test_df["id"].astype(str).tolist()
    for i, qid in enumerate(qids):
        doc_ids = [str(x) for x in pred_map[qid]]
        category = "" if phase == 1 else str(category_preds[i])
        rows.append({
            "query_id": qid,
            "relevant_doc_ids": json.dumps(doc_ids),
            "category": category,
        })
    out = pd.DataFrame(rows, columns=["query_id", "relevant_doc_ids", "category"])
    out.to_csv(output_path, index=False)
    return out

submission_path = Path(CFG.output_dir) / "solutions_SeaFour.csv"

if "final_pred_map" in globals():
    submission_df = write_submission(
        queries_test_df=queries_test_df,
        pred_map=final_pred_map,
        category_preds=final_test_categories,
        output_path=submission_path,
        phase=CFG.phase,
    )
    display(submission_df.head())
    print("Saved:", submission_path.resolve())
else:
    print("Run final prediction first.")

"""
## Additional improvement ideas

The notebook now covers the requested core workflow, but there are still reasonable extensions:

1. **Top-k optimization**
   - Try `k in {20, 50, 100, 150}` on the validation split.
   - Higher `k` may raise recall but can lower precision.

2. **Field-weight tuning**
   - Increase `title_weight` or `tags_weight` if short fields are strongly predictive.

3. **More embedding models**
   - Compare compact retrieval models first.
   - Only move to larger models if the validation gain is real.

4. **Ablation study**
   - Compare with and without tags.
   - Compare with and without category bonus.
   - Compare with and without dense retrieval.

5. **Cross-validation**
   - Replace one train/validation split with repeated stratified validation for more stable estimates.
"""

"""
# Final report — what was improved and why

## 1. The original notebook was converted from a submission script into a full IR workflow

The original version was mostly a direct pipeline from loaded data to CSV export.  
That is useful for final submission, but it is not enough for controlled improvement.  
The new notebook adds:
- a clean configuration section,
- explicit train/validation splitting,
- local metric computation,
- systematic experiment tracking,
- a final report section.

This changes the project from **single-run code** into a **reproducible retrieval experiment notebook**.

## 2. BM25Okapi was replaced by BM25+

The competition request explicitly mentions **BM25+**.  
BM25+ is preferable here because it reduces some of the length-normalization bias of basic BM25 and is a better fit when document lengths vary substantially.  
The notebook now uses `rank_bm25.BM25Plus` with configurable `k1`, `b`, and `delta`.

## 3. Local validation was added using `queries_train.json` and `qgts_train.json`

A major limitation of the original notebook is that it does not properly measure improvement before submission.  
The new notebook builds a gold table from `qgts_train`, merges it with the training queries, and evaluates:
- Recall@k
- Precision@k
- MRR@k
- Accuracy

This is essential because the leaderboard score is a weighted combination, so optimization should happen locally before generating a submission.

## 4. Retrieval was expanded from isolated baselines to cooperative models

The new notebook supports:
- TF-IDF
- BM25+
- dense embedding retrieval
- reciprocal rank fusion (RRF)
- category-aware reranking

This is a better match for the competition goal because IR systems often perform better when lexical and semantic retrieval complement each other instead of competing independently.

## 5. Query classification was upgraded into a two-stage Phase 2 system

The notebook now contains a Phase 2 classifier:
- **first stage:** deterministic tag-based category prediction
- **second stage:** TF-IDF + Logistic Regression fallback over text

This is a strong design because it uses structured metadata when available and falls back to learned text features otherwise.

## 6. Classification now helps retrieval

Instead of predicting categories only for the submission column, the notebook uses the predicted category to help ranking.  
Documents receive a small score bonus when their inferred pseudo-category matches the predicted query category.

This is a controlled reranking strategy:
- it exploits classifier information,
- it does not hard-filter the candidate set,
- it reduces the risk of catastrophic false exclusions.

## 7. Document pseudo-categories were introduced

If document tags are available in `docs.json`, the notebook infers a pseudo-category from them using the training tag-to-category mapping.  
This creates a bridge between Phase 2 classification and retrieval optimization.

## 8. Caching was added for expensive dense steps

Dense embeddings are cached to disk.  
This matters because document encoding is usually the most expensive part of notebook reruns.  
With caching, the workflow becomes iterative instead of slow and frustrating.

## 9. Submission generation was corrected and clarified

The notebook now explicitly writes:
- empty category strings for Phase 1,
- predicted category labels for Phase 2,
- JSON string lists for the retrieved document IDs.

This removes ambiguity and makes the final output consistent with the required submission structure.

## 10. Recommended final workflow for SeaFour

1. Run the notebook once to inspect the data and verify tag/category purity.
2. Run the validation experiments and compare:
   - TF-IDF
   - BM25+
   - dense models
   - fusion
   - fusion + category bonus
3. Select the best dense model based on validation score.
4. Retrain on all training queries.
5. Generate:
   - one Phase 1 submission,
   - one Phase 2 submission.
6. Keep the experiment table and this report in the notebook as part of the project documentation.

## 11. Limitations and honest notes

- The uploaded chat files do **not** include `docs.json`, so full retrieval execution must be done in Kaggle or with a local copy of the full competition corpus.
- The best final dense model cannot be declared in advance without running the validation table on the real corpus.
- The category-aware reranking depends on document tags being available and informative.

## 12. Short conclusion

The improved notebook is now aligned with the competition structure:
- Phase 1: strong lexical and semantic retrieval baselines
- Phase 2: classification integrated into reranking
- reproducible local evaluation
- final submission generation
- documented improvement process for reporting
"""
