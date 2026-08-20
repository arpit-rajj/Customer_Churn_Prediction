import os
import joblib
import logging
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)

def train_baselines(X_train, y_train, preprocessor) -> Dict[str, Pipeline]:
    """Trains baseline models for comparison."""
    logger.info("Training baselines...")
    
    baselines = {
        "Dummy": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DummyClassifier(strategy='prior'))]),
        "LogisticRegression": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))]),
        "DecisionTree": Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=5))])
    }
    
    for name, pipeline in baselines.items():
        logger.info(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
    return baselines

def train_random_forest(X_train, y_train, preprocessor, tune: bool = True) -> Pipeline:
    """Trains and optionally tunes a Random Forest model."""
    logger.info("Training Random Forest...")
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    
    if tune:
        logger.info("Tuning hyperparameters using RandomizedSearchCV...")
        # Note: the parameters need to be prefixed with 'classifier__' because of the Pipeline
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2']
        }
        
        # Use stratified CV and optimize for recall since that's our primary metric
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, 
            n_iter=10, cv=3, scoring='recall', 
            random_state=42, n_jobs=-1, verbose=1
        )
        
        search.fit(X_train, y_train)
        logger.info(f"Best parameters found: {search.best_params_}")
        return search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        return pipeline

def save_model(model: Pipeline, filepath: str):
    """Saves the trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    
def load_model(filepath: str) -> Pipeline:
    """Loads a model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at {filepath}")
    return joblib.load(filepath)
