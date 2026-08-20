import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from src.models.evaluate import calculate_business_cost, evaluate_model
import logging

logger = logging.getLogger(__name__)

def optimize_threshold(y_val: np.ndarray, y_val_proba: np.ndarray, 
                       thresholds: List[float] = None, 
                       fn_cost: float = 1000.0, fp_cost: float = 50.0) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates multiple thresholds on the validation set and selects the optimal one based on business cost.
    Returns the optimal threshold and a dictionary of metrics for that threshold.
    """
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        
    best_threshold = 0.5
    min_cost = float('inf')
    best_metrics = {}
    
    logger.info("Evaluating thresholds:")
    logger.info(f"{'Threshold':<10} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'Cost':<10}")
    
    for thresh in thresholds:
        y_pred_thresh = (y_val_proba >= thresh).astype(int)
        metrics = evaluate_model(y_val, y_pred_thresh, y_val_proba)
        cost = calculate_business_cost(y_val, y_pred_thresh, fn_cost=fn_cost, fp_cost=fp_cost)
        
        logger.info(f"{thresh:<10.2f} | {metrics['recall']:<8.4f} | {metrics['precision']:<10.4f} | {metrics['f1']:<8.4f} | ${cost:<10.2f}")
        
        if cost < min_cost:
            min_cost = cost
            best_threshold = thresh
            best_metrics = metrics
            
    logger.info(f"Optimal threshold selected: {best_threshold} with business cost ${min_cost:.2f}")
    
    return best_threshold, best_metrics
