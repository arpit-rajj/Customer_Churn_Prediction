from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.sql import func
from src.db.connection import Base

class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True, index=True)
    surname = Column(String)
    credit_score = Column(Integer)
    geography = Column(String)
    gender = Column(String)
    age = Column(Integer)
    tenure = Column(Integer)
    balance = Column(Float)
    num_of_products = Column(Integer)
    has_cr_card = Column(Integer)
    is_active_member = Column(Integer)
    estimated_salary = Column(Float)
    exited = Column(Integer) # Target variable
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"))
    model_version = Column(String, index=True)
    churn_probability = Column(Float)
    predicted_churn = Column(Boolean)
    risk_level = Column(String) # LOW, MEDIUM, HIGH
    prediction_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Index for fast querying by customer
    __table_args__ = (
        Index('idx_pred_customer', 'customer_id'),
        Index('idx_pred_churn', 'predicted_churn'),
    )

class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    model_version = Column(String, primary_key=True, index=True)
    model_type = Column(String)
    training_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    training_rows = Column(Integer)
    recall = Column(Float)
    precision = Column(Float)
    f1 = Column(Float)
    roc_auc = Column(Float)
    pr_auc = Column(Float)
    threshold = Column(Float)
    artifact_path = Column(String)
