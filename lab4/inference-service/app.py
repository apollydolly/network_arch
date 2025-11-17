import mlflow
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "https://mlflow.labs.itmo.loc"
MODEL_NAME = "Lab4-Classification-Model"
MODEL_STAGE = "Production"

print(f"Using MLflow at: {MLFLOW_TRACKING_URI}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

app = FastAPI(
    title="ML Inference Service",
    description="Сервис для предсказаний с загрузкой моделей из MLflow Registry",
    version="1.0.0"
)

model = None

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str = "Lab4-Classification-Model"
    model_source: str = "MLflow Registry"
    
    class Config:
        protected_namespaces = ()

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Starting Inference Service...")
    logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"Loading model: {MODEL_NAME} (Stage: {MODEL_STAGE})")
    
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Model URI: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info("Model loaded successfully from MLflow Registry!")
        logger.info(f"Model type: {type(model)}")
        
        if hasattr(model, 'n_features_in_'):
            logger.info(f"Model expects {model.n_features_in_} features")
        if hasattr(model, 'n_estimators'):
            logger.info(f"Model has {model.n_estimators} estimators")
        
    except Exception as e:
        logger.error(f"Failed to load model from MLflow Registry: {e}")
        raise e

@app.get("/")
async def root():
    return {
        "service": "ML Inference Service", 
        "status": "running",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "model_source": "MLflow Model Registry"
    }

@app.get("/health")
async def health():
    status = "healthy" if model is not None else "degraded"
    return {
        "status": status,
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "mlflow_uri": mlflow.get_tracking_uri()
    }

@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_type": type(model).name,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "source": "MLflow Model Registry",
        "run_id": "1750b23f64854374ab838f4e967cbec6"
    }
    
    if hasattr(model, 'n_features_in_'):
        info["n_features"] = model.n_features_in_
    if hasattr(model, 'n_estimators'):
        info["n_estimators"] = model.n_estimators
    if hasattr(model, 'feature_names_in_'):
        info["feature_names"] = model.feature_names_in_.tolist()
        
    return info

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_array = np.array([request.features])
        if hasattr(model, 'n_features_in_') and len(request.features) != model.n_features_in_:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {model.n_features_in_} features, got {len(request.features)}"
            )

        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array).max()
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=float(probability),
            model_name=MODEL_NAME,
            model_source="MLflow Registry"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/example")
async def example_prediction():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
    else:
        n_features = 20  # fallback

    example_features = [0.1] * n_features
    
    features_array = np.array([example_features])
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array).max()
        
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability),
        "features_used": n_features,
        "model_name": MODEL_NAME,
        "model_source": "MLflow Registry"
    }
