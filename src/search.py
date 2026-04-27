# search.py
"""
Implements hybrid search using BM25 (sparse) and vector search (dense).
"""

from rank_bm25 import BM25Okapi
from config import SPARSE_WEIGHT, DENSE_WEIGHT, COLLECTION_NAME

def init_bm25(df):
    tokenized_corpus = [str(text).split() for text in df["combined_text"]]
    return BM25Okapi(tokenized_corpus)

def hybrid_search(client, bm25, model, query):
    if model is None or client is None:
        bm25_scores = bm25.get_scores(query.split())
        ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:10]
        return ranked

    query_vector = model.encode([query])[0]
    dense_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=10
    )

    bm25_scores = bm25.get_scores(query.split())
    hybrid_scores = {}

    # Combine scores
    for result in dense_results:
        hybrid_scores[result.id] = DENSE_WEIGHT * result.score

    for idx, score in enumerate(bm25_scores):
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + SPARSE_WEIGHT * score

    # Sort by combined score
    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    return ranked
