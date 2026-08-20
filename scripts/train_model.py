import argparse
import logging
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from src.db.connection import engine
from src.db.repository import save_model_metadata
from src.db.models import ModelMetadata
from sqlalchemy.orm import Session
from src.data.preprocessing import get_preprocessing_pipeline
from src.models.train import train_baselines, train_random_forest, save_model
from src.models.threshold import optimize_threshold
from src.models.evaluate import evaluate_model
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=settings.MODEL_VERSION, help="Model version to save")
    args = parser.parse_args()
    
    logger.info("Loading data from PostgreSQL...")
    df = pd.read_sql("SELECT * FROM customers", engine)
    
    if df.empty:
        raise ValueError("No data found in PostgreSQL. Run ingestion script first.")
        
    logger.info(f"Loaded {len(df)} rows from PostgreSQL.")
    
    # Check Class Imbalance
    class_counts = df['exited'].value_counts()
    logger.info("Class Imbalance Analysis:")
    logger.info(f"Class 0 (Retained): {class_counts.get(0, 0)} ({class_counts.get(0, 0) / len(df) * 100:.2f}%)")
    logger.info(f"Class 1 (Churned): {class_counts.get(1, 0)} ({class_counts.get(1, 0) / len(df) * 100:.2f}%)")
    
    # Features and Target
    X = df.drop(columns=['exited', 'customer_id', 'surname', 'created_at'])
    y = df['exited'].values
    
    # Train/Validation/Test Split (60/20/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
    ) # 0.25 * 0.8 = 0.2
    
    preprocessor = get_preprocessing_pipeline()
    
    # Baselines
    baselines = train_baselines(X_train, y_train, preprocessor)
    logger.info("Baseline performances on Validation set:")
    for name, model in baselines.items():
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model(y_val, y_val_pred, y_val_proba)
        logger.info(f"{name} - Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}, F1: {metrics['f1']:.4f}")
        
    # Main Model
    best_rf = train_random_forest(X_train, y_train, preprocessor, tune=True)
    
    # Threshold optimization on Validation Set
    logger.info("Optimizing threshold on validation set...")
    y_val_proba_rf = best_rf.predict_proba(X_val)[:, 1]
    optimal_threshold, _ = optimize_threshold(y_val, y_val_proba_rf)
    
    # Final Evaluation on Test Set
    logger.info("Evaluating on untouched TEST set with optimal threshold...")
    y_test_proba = best_rf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba)
    
    logger.info(f"Final Test Metrics:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: \n{v}")
            
    # Save Model Artifact
    artifact_path = os.path.join(settings.MODEL_DIR, f"{args.version}.pkl")
    save_model(best_rf, artifact_path)
    
    # Save Metadata to DB
    logger.info("Saving model metadata to database...")
    metadata = {
        "model_version": args.version,
        "model_type": "RandomForestClassifier",
        "training_rows": len(X_train),
        "recall": test_metrics["recall"],
        "precision": test_metrics["precision"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "threshold": optimal_threshold,
        "artifact_path": artifact_path
    }
    
    with Session(engine) as session:
        save_model_metadata(session, metadata)
        
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
