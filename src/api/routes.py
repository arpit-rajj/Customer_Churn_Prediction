from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import os

from src.db.connection import get_db
from src.db.repository import save_prediction, get_model_metadata
from src.api.schemas import PredictionRequest, PredictionResponse
from src.models.predict import predict_churn
from src.models.train import load_model
from src.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global variables to hold the loaded model and metadata in memory
ML_MODEL = None
MODEL_METADATA = None

def load_ml_model(db: Session):
    """Loads the model and its metadata into memory if not already loaded."""
    global ML_MODEL, MODEL_METADATA
    
    if ML_MODEL is None or MODEL_METADATA is None:
        logger.info(f"Loading model version {settings.MODEL_VERSION} into memory...")
        
        # Load metadata from DB
        MODEL_METADATA = get_model_metadata(db, settings.MODEL_VERSION)
        if not MODEL_METADATA:
            raise HTTPException(status_code=500, detail=f"Model metadata for {settings.MODEL_VERSION} not found in database.")
            
        # Load model artifact
        artifact_path = MODEL_METADATA.artifact_path
        if not os.path.exists(artifact_path):
            raise HTTPException(status_code=500, detail=f"Model artifact not found at {artifact_path}")
            
        ML_MODEL = load_model(artifact_path)
        logger.info("Model loaded successfully.")

@router.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy"}

@router.get("/model/info")
def model_info(db: Session = Depends(get_db)):
    """Returns information about the currently loaded model."""
    load_ml_model(db)
    return {
        "model_version": MODEL_METADATA.model_version,
        "model_type": MODEL_METADATA.model_type,
        "training_timestamp": MODEL_METADATA.training_timestamp,
        "metrics": {
            "recall": MODEL_METADATA.recall,
            "precision": MODEL_METADATA.precision,
            "f1": MODEL_METADATA.f1,
            "roc_auc": MODEL_METADATA.roc_auc,
            "pr_auc": MODEL_METADATA.pr_auc,
        },
        "threshold": MODEL_METADATA.threshold
    }

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Makes a churn prediction for a customer.
    Persists the prediction in the database.
    """
    load_ml_model(db)
    
    customer_data = request.dict()
    
    try:
        # Predict
        prob, pred_class, risk = predict_churn(
            model=ML_MODEL, 
            customer_data=customer_data, 
            threshold=MODEL_METADATA.threshold
        )
        
        # Prepare prediction record
        prediction_record = {
            "customer_id": request.customer_id,
            "model_version": MODEL_METADATA.model_version,
            "churn_probability": prob,
            "predicted_churn": pred_class,
            "risk_level": risk
        }
        
        # Save to DB
        save_prediction(db, prediction_record)
        
        # Return response
        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=prob,
            predicted_churn=pred_class,
            risk_level=risk,
            model_version=MODEL_METADATA.model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")
