import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add engineered features.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_eng = X.copy()
        
        # Balance per product
        # Rationale: Customers with high balance but few products might be different from those with many products.
        # Leakage risk: None (uses only static features).
        X_eng['BalancePerProduct'] = X_eng['Balance'] / X_eng['NumOfProducts'].replace(0, 1) # avoid division by zero
        
        # IsZeroBalance
        # Rationale: Customers with 0 balance might behave distinctly.
        # Leakage risk: None.
        X_eng['IsZeroBalance'] = (X_eng['Balance'] == 0).astype(int)
        
        # Products per tenure
        # Rationale: Measures how fast the customer acquires products.
        # Leakage risk: None.
        X_eng['ProductsPerTenure'] = X_eng['NumOfProducts'] / (X_eng['Tenure'] + 1) # Add 1 to avoid division by zero
        
        # Age group
        # Rationale: Age may have non-linear effects; grouping might help linear models (less so for RF, but good practice to explore).
        # Leakage risk: None.
        # We will keep Age as continuous, but create an interactive feature Age*IsActiveMember
        X_eng['ActiveByAge'] = X_eng['IsActiveMember'] * X_eng['Age']
        
        return X_eng
