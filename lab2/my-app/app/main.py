from fastapi import FastAPI, HTTPException
from app.redis_client import get_cached_data, set_cached_data
from app.qdrant_client import upsert_vector, search_vectors
from app.models import Item, SearchResponse
import requests
from pydantic import BaseModel

app = FastAPI(title="Vector Search App")

class InferenceRequest(BaseModel):
    features: list[float]

class InferenceResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str

INFERENCE_SERVICE_URL = "http://inference-service.default.svc.cluster.local:8000"

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
        
@app.post("/ml-predict/")
async def ml_predict(request: InferenceRequest):
    """Предсказание через ML модель"""
    try:
        response = requests.post(
            f"{INFERENCE_SERVICE_URL}/predict",
            json={"features": request.features},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Inference service error: {str(e)}"
        )

@app.get("/ml-health/")
async def ml_health():
    """Проверка здоровья ML сервиса"""
    try:
        response = requests.get(f"{INFERENCE_SERVICE_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference service unavailable: {str(e)}"
        )

@app.get("/ml-example/")
async def ml_example():
    """Пример предсказания"""
    try:
        response = requests.get(f"{INFERENCE_SERVICE_URL}/example", timeout=5)
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference service unavailable: {str(e)}"
        )
