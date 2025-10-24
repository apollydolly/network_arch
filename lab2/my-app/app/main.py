from fastapi import FastAPI, HTTPException
from app.redis_client import get_cached_data, set_cached_data
from app.qdrant_client import upsert_vector, search_vectors
from app.models import Item, SearchResponse

app = FastAPI(title="Vector Search App")

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/items/")
def add_item(item: Item):
    try:
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=item.id,
            vector=item.vector,
            payload=item.payload,
        )
        result = upsert_vector(point)
        set_cached_data(f"item:{item.id}", "inserted")
        return {"status": "added", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding item: {str(e)}")

@app.get("/search/")
def search_vectors_endpoint(vector: str, limit: int = 5):
    try:
        vector_list = [float(x) for x in vector.split(",")]
        results = search_vectors(vector_list, limit)
        return [
            SearchResponse(
                id=hit.id,
                score=hit.score,
                payload=hit.payload,
            )
            for hit in results
        ]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid vector format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/cache/{key}")
def get_cache(key: str):
    try:
        value = get_cached_data(key)
        return {"key": key, "value": value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")

@app.post("/cache/{key}")
def set_cache(key: str, value: str):
    try:
        set_cached_data(key, value)
        return {"status": "cached"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache error: {str(e)}")
