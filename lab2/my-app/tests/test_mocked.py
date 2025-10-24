import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Импортируем приложение
from app.main import app

client = TestClient(app)

# Мокаем зависимости на уровне модулей, а не app.main
@patch('app.redis_client.redis_client')
@patch('app.qdrant_client.qdrant_client')
def test_health_check(mock_qdrant, mock_redis):
    """Тест проверки работоспособности"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

@patch('app.redis_client.redis_client')
@patch('app.qdrant_client.qdrant_client')
def test_add_item(mock_qdrant, mock_redis):
    """Тест добавления вектора с моками"""
    # Настраиваем моки
    mock_redis.setex.return_value = True
    mock_qdrant.upsert.return_value = {"status": "ok"}
    
    item_data = {
        "id": "test_item_1",
        "vector": [0.1, 0.2, 0.3, 0.4] * 32,
        "payload": {"text": "test document"}
    }
    
    response = client.post("/items/", json=item_data)
    assert response.status_code == 200
    assert "added" in response.json()["status"]

@patch('app.redis_client.redis_client')
@patch('app.qdrant_client.qdrant_client')
def test_search_vectors(mock_qdrant, mock_redis):
    """Тест поиска векторов с моками"""
    # Создаем мок для результата поиска
    mock_result = Mock()
    mock_result.id = "test_item_1"
    mock_result.score = 0.95
    mock_result.payload = {"text": "test document"}
    
    mock_qdrant.search.return_value = [mock_result]
    
    response = client.get("/search/?vector=0.1,0.2,0.3,0.4&limit=5")
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert results[0]["id"] == "test_item_1"

@patch('app.redis_client.redis_client')
@patch('app.qdrant_client.qdrant_client')
def test_cache_operations(mock_qdrant, mock_redis):
    """Тест работы с кешем с моками"""
    # Настраиваем мок Redis
    mock_redis.get.return_value = "test_value"
    mock_redis.setex.return_value = True
    
    # Записываем в кеш
    response = client.post("/cache/test_key", params={"value": "test_value"})
    assert response.status_code == 200
    assert response.json() == {"status": "cached"}
    
    # Читаем из кеша
    response = client.get("/cache/test_key")
    assert response.status_code == 200
    assert response.json()["value"] == "test_value"

@patch('app.redis_client.redis_client')
@patch('app.qdrant_client.qdrant_client')
def test_invalid_vector(mock_qdrant, mock_redis):
    """Тест обработки невалидного вектора"""
    response = client.get("/search/?vector=invalid,vector")
    assert response.status_code == 422
    assert "Invalid vector format" in response.json()["detail"]

@patch('app.redis_client.redis_client')
@patch('app.qdrant_client.qdrant_client')
def test_empty_vector(mock_qdrant, mock_redis):
    """Тест обработки пустого вектора"""
    response = client.get("/search/?vector=")
    assert response.status_code == 422

def test_documentation():
    """Тест доступности документации (не требует моков)"""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/redoc")
    assert response.status_code == 200

def test_health_without_mocks():
    """Тест health check без моков (должен работать)"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}
