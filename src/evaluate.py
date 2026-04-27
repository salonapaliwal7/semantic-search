# evaluate.py
"""
Implements retrieval evaluation metrics like Precision@K, Recall@K, and MRR.
"""

import numpy as np

def precision_at_k(results, relevant, k=10):
    retrieved = results[:k]
    return len(set(retrieved) & set(relevant)) / k

def recall_at_k(results, relevant, k=10):
    retrieved = results[:k]
    return len(set(retrieved) & set(relevant)) / len(relevant)

def mean_reciprocal_rank(all_results, all_relevant):
    rr = []
    for results, relevant in zip(all_results, all_relevant):
        rank = next((i+1 for i, r in enumerate(results) if r in relevant), 0)
        rr.append(1/rank if rank else 0)
    return np.mean(rr)
