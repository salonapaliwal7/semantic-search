# app.py
"""
Streamlit interface for testing the hybrid search system.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from config import DATA_PATH
from preprocess import preprocess_data
from search import init_bm25, hybrid_search
from datasets import load_dataset



st.title("🛍️ Hybrid Semantic Search - Product Catalog")

# Keep Hugging Face cache local to project to avoid home directory permission issues.
project_root = Path(__file__).resolve().parent.parent
hf_cache_dir = project_root / ".hf_cache"
hf_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)

try:
    df = load_dataset("naga-jay/amazon-laptop-product-catalog", cache_dir=str(hf_cache_dir))
    st.caption("Data source: Hugging Face dataset")
except Exception as exc:
    local_csv_path = project_root / DATA_PATH
    st.warning(f"Hugging Face dataset unavailable ({type(exc).__name__}). Falling back to local CSV.")
    df = pd.read_csv(local_csv_path)
    st.caption(f"Data source: local file `{local_csv_path}`")

df = preprocess_data(df)

bm25 = init_bm25(df)
st.caption("Search mode: BM25 only")

query = st.text_input("Enter your search query:")
if query:
    results = hybrid_search(None, bm25, None, query)
    st.write("### Top Results")
    for idx, score in results:
        row = df.iloc[idx]
        title = row.get("title", row.get("name", "Untitled"))
        brand = row.get("brand", "Unknown")
        category = row.get("main_category", row.get("category", "Unknown"))
        st.write(f"**{title}** — {brand} ({category}) — Score: {round(score,3)}")
