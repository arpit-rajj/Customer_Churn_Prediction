import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    'RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
    'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'Exited'
]

def validate_raw_data(df: pd.DataFrame) -> dict:
    """
    Validates raw churn dataset.
    Returns a dictionary with validation status and errors.
    """
    errors = []
    
    # 1. Check missing columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    # 2. Check missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        cols_with_missing = missing_values[missing_values > 0].index.tolist()
        errors.append(f"Missing values found in columns: {cols_with_missing}")

    # 3. Check duplicate CustomerIds
    if 'CustomerId' in df.columns:
        duplicates = df['CustomerId'].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate CustomerIds.")

    # 4. Check data ranges
    if 'CreditScore' in df.columns:
        if df['CreditScore'].min() < 300 or df['CreditScore'].max() > 850:
            errors.append("CreditScore out of expected range [300, 850].")
            
    if 'Age' in df.columns:
        if df['Age'].min() < 18 or df['Age'].max() > 100:
            errors.append(f"Age contains unusual values: min={df['Age'].min()}, max={df['Age'].max()}.")
            
    if 'Tenure' in df.columns:
        if df['Tenure'].min() < 0 or df['Tenure'].max() > 10:
            errors.append("Tenure out of expected range [0, 10].")
            
    if 'NumOfProducts' in df.columns:
        if df['NumOfProducts'].min() < 1 or df['NumOfProducts'].max() > 4:
            errors.append("NumOfProducts out of expected range [1, 4].")

    if 'HasCrCard' in df.columns:
        if not set(df['HasCrCard'].unique()).issubset({0, 1}):
            errors.append("HasCrCard must be 0 or 1.")
            
    if 'IsActiveMember' in df.columns:
        if not set(df['IsActiveMember'].unique()).issubset({0, 1}):
            errors.append("IsActiveMember must be 0 or 1.")

    if 'Exited' in df.columns:
        if not set(df['Exited'].unique()).issubset({0, 1}):
            errors.append("Exited must be 0 or 1.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }
