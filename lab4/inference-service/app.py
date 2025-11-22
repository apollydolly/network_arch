import mlflow
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
from typing import List
import urllib3
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi import Response
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

app = FastAPI(
    title="ML Inference Service",
    description="Сервис для предсказаний с моделью из MLflow",
    version="1.0.0"
)

MODEL_PATH = "/app/model"
model = None
PREDICTION_COUNTER = Counter('inference_predictions_total', 'Total prediction requests', ['status'])
PREDICTION_DURATION = Histogram('inference_prediction_duration_seconds', 'Prediction request duration')

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str = "Lab4-Simple-Final"
    
    class Config:
        protected_namespaces = ()

@app.on_event("startup")
async def startup_event():
    global model
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Inference Service...")
    logger.info(f"Loading model from: {MODEL_PATH}")
    
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
  
        if hasattr(model, 'n_features_in_'):
            logger.info(f"Model expects {model.n_features_in_} features")
        else:
            logger.info("Model n_features_in_ attribute not found, assuming 20 features")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.get("/")
async def root():
    return {
        "service": "ML Inference Service", 
        "status": "running",
        "model_loaded": model is not None,
        "model_name": "Lab4-Simple-Final"
    }

@app.get("/health")
async def health():
    status = "healthy" if model is not None else "degraded"
    return {
        "status": status,
        "model_loaded": model is not None
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(REGISTRY), media_type="text/plain")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    if model is None:
        PREDICTION_COUNTER.labels(status='error').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 20
        
        if len(request.features) != expected_features:
            PREDICTION_COUNTER.labels(status='error').inc()
            PREDICTION_DURATION.observe(time.time() - start_time)
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {expected_features} features, got {len(request.features)}"
            )
        
        features_array = np.array([request.features])
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array).max()
        
        PREDICTION_COUNTER.labels(status='success').inc()
        PREDICTION_DURATION.observe(time.time() - start_time)
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=float(probability)
        )
        
    except HTTPException:
        PREDICTION_DURATION.observe(time.time() - start_time)
        raise
    except Exception as e:
        PREDICTION_COUNTER.labels(status='error').inc()
        PREDICTION_DURATION.observe(time.time() - start_time)
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/example")
async def example_prediction():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    example_features = [0.1] * 20
    features_array = np.array([example_features])
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array).max()
        
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability),
        "features_used": 20,
        "model_name": "Lab4-Simple-Final"
    }
