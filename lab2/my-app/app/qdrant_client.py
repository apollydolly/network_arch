from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import Config

qdrant_client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)

#qdrant_client = QdrantClient(host="qdrant-service", port=6333)

#создаём коллекцию (если ещё не создана)
try:
    qdrant_client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=128, distance=Distance.COSINE),
    )
except Exception as e:
    print(f"Collection already exists: {e}")

def upsert_vector(point: PointStruct):
    return qdrant_client.upsert(
        collection_name="test_collection",
        wait=True,
        points=[point],
    )

def search_vectors(vector: list[float], limit: int = 5):
    return qdrant_client.search(
        collection_name="test_collection",
        query_vector=vector,
        limit=limit,
    )
