import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, Any

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates model performance and returns a dictionary of metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics

def calculate_business_cost(y_true: np.ndarray, y_pred: np.ndarray, fn_cost: float = 1000.0, fp_cost: float = 50.0) -> float:
    """
    Calculates the hypothetical business cost of the model's predictions.
    
    False Negative Cost (fn_cost): The cost of missing a true churner (e.g., lost customer lifetime value).
    False Positive Cost (fp_cost): The cost of incorrectly predicting churn (e.g., cost of a retention campaign).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_cost = (fn * fn_cost) + (fp * fp_cost)
    return float(total_cost)
