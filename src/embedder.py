# embedder.py
"""
Generates dense vector embeddings using Sentence Transformers.
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)
