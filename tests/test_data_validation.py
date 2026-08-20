import pandas as pd
import pytest
from src.data.validation import validate_raw_data

def test_missing_columns():
    df = pd.DataFrame({"Age": [20, 30]})
    result = validate_raw_data(df)
    assert not result["is_valid"]
    assert any("Missing expected columns" in err for err in result["errors"])

def test_invalid_age_range():
    df = pd.DataFrame({
        'RowNumber': [1], 'CustomerId': [1], 'Surname': ['A'], 'CreditScore': [500],
        'Geography': ['France'], 'Gender': ['Male'], 'Age': [200], # Invalid
        'Tenure': [5], 'Balance': [1000], 'NumOfProducts': [1], 'HasCrCard': [1],
        'IsActiveMember': [1], 'EstimatedSalary': [50000], 'Exited': [0]
    })
    result = validate_raw_data(df)
    assert not result["is_valid"]
    assert any("Age contains unusual values" in err for err in result["errors"])

def test_valid_data():
    df = pd.DataFrame({
        'RowNumber': [1], 'CustomerId': [1], 'Surname': ['A'], 'CreditScore': [500],
        'Geography': ['France'], 'Gender': ['Male'], 'Age': [40],
        'Tenure': [5], 'Balance': [1000], 'NumOfProducts': [1], 'HasCrCard': [1],
        'IsActiveMember': [1], 'EstimatedSalary': [50000], 'Exited': [0]
    })
    result = validate_raw_data(df)
    assert result["is_valid"]
