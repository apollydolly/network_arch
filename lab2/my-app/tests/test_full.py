from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Тест проверки работоспособности"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_documentation():
    """Тест доступности документации"""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/redoc")
    assert response.status_code == 200

def test_api_schema():
    """Тест доступности OpenAPI схемы"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "openapi" in response.json()

def test_invalid_endpoint():
    """Тест обработки несуществующего эндпоинта"""
    response = client.get("/nonexistent")
    assert response.status_code == 404

def test_invalid_vector_format():
    """Тест обработки невалидного формата вектора"""
    response = client.get("/search/?vector=invalid")
    assert response.status_code == 422

def test_cache_endpoints_exist():
    """Тест что эндпоинты кеша существуют"""
    # Просто проверяем что эндпоинты отвечают (даже с ошибкой)
    response = client.get("/cache/test")
    # Может быть 500 из-за недоступности Redis, но не 404
    assert response.status_code != 404
    
    response = client.post("/cache/test", params={"value": "test"})
    assert response.status_code != 404
