import numpy as np
from src.models.evaluate import evaluate_model
from src.models.predict import determine_risk_level

def test_evaluate_model():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.7])
    
    metrics = evaluate_model(y_true, y_pred, y_prob)
    
    assert metrics['accuracy'] == 0.75
    assert metrics['recall'] == 1.0 # True positives: 2, False negatives: 0
    assert metrics['precision'] == 2/3 # True positives: 2, False positives: 1

def test_determine_risk_level():
    threshold = 0.5
    assert determine_risk_level(0.6, threshold) == "HIGH"
    assert determine_risk_level(0.5, threshold) == "HIGH"
    assert determine_risk_level(0.3, threshold) == "MEDIUM" # >= 0.25
    assert determine_risk_level(0.2, threshold) == "LOW"
