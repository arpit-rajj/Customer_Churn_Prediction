import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

def determine_risk_level(prob: float, threshold: float) -> str:
    """
    Determines risk level based on probability.
    Rules:
    - LOW: Probability < 0.5 * threshold
    - MEDIUM: 0.5 * threshold <= Probability < threshold
    - HIGH: Probability >= threshold
    """
    if prob >= threshold:
        return "HIGH"
    elif prob >= (0.5 * threshold):
        return "MEDIUM"
    else:
        return "LOW"

def predict_churn(model: Pipeline, customer_data: Dict[str, Any], threshold: float) -> Tuple[float, bool, str]:
    """
    Predicts churn for a single customer.
    Returns (probability, predicted_class, risk_level).
    """
    # Convert dict to DataFrame as the pipeline expects a DataFrame
    df = pd.DataFrame([customer_data])
    
    # Get probability of class 1 (Churn)
    prob = float(model.predict_proba(df)[0, 1])
    
    # Apply custom threshold
    pred_class = bool(prob >= threshold)
    
    # Get risk level
    risk = determine_risk_level(prob, threshold)
    
    return prob, pred_class, risk
