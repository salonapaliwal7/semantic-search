# vector_db.py
"""
Handles vector database operations (Qdrant setup, upsert, search).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from config import COLLECTION_NAME, QDRANT_URL

def connect_qdrant():
    return QdrantClient(QDRANT_URL)

def create_collection(client, dim):
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": dim, "distance": "Cosine"}
    )

def upload_points(client, embeddings, df):
    points = []
    for i, row in df.iterrows():
        points.append(
            PointStruct(
                id=int(i),
                vector=embeddings[i].tolist(),
                payload={
                    "title": row.get("name"),
                    "brand": row.get("brand"),
                    "category": row.get("main_category"),
                }
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Uploaded {len(points)} points to Qdrant.")
