from pydantic import BaseModel

class Item(BaseModel):
    id: str
    vector: list[float]
    payload: dict = None

class SearchResponse(BaseModel):
    id: str
    score: float
    payload: dict = None
