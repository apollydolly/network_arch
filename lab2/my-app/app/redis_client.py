import redis
from app.config import Config

redis_client = redis.Redis(
    host=Config.REDIS_HOST, 
    port=Config.REDIS_PORT, 
    decode_responses=True
)
#redis_client = redis.Redis(host='redis-service', port=6379, decode_responses=True)

def get_cached_data(key: str):
    return redis_client.get(key)

def set_cached_data(key: str, value: str, expire: int = 300):
    redis_client.setex(key, expire, value)
