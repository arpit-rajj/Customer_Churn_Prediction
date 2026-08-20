from sqlalchemy.orm import Session
from src.db import models
from typing import List, Optional

def create_customer(db: Session, customer_data: dict) -> models.Customer:
    db_customer = models.Customer(**customer_data)
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    return db_customer

def get_customer(db: Session, customer_id: int) -> Optional[models.Customer]:
    return db.query(models.Customer).filter(models.Customer.customer_id == customer_id).first()

def save_prediction(db: Session, prediction_data: dict) -> models.Prediction:
    db_pred = models.Prediction(**prediction_data)
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

def get_predictions_by_customer(db: Session, customer_id: int, limit: int = 10) -> List[models.Prediction]:
    return db.query(models.Prediction).filter(
        models.Prediction.customer_id == customer_id
    ).order_by(models.Prediction.prediction_timestamp.desc()).limit(limit).all()

def save_model_metadata(db: Session, metadata: dict) -> models.ModelMetadata:
    db_meta = models.ModelMetadata(**metadata)
    # Use merge to update if version exists
    db_meta = db.merge(db_meta)
    db.commit()
    return db_meta

def get_model_metadata(db: Session, model_version: str) -> Optional[models.ModelMetadata]:
    return db.query(models.ModelMetadata).filter(models.ModelMetadata.model_version == model_version).first()
