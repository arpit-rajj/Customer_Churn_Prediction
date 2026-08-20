import pandas as pd
from typing import List, Dict, Any
import logging
from src.data.validation import validate_raw_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from CSV and validates it.
    Raises ValueError if validation fails.
    """
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    logger.info("Validating data...")
    validation_result = validate_raw_data(df)
    
    if not validation_result["is_valid"]:
        logger.error("Data validation failed!")
        for err in validation_result["errors"]:
            logger.error(err)
        raise ValueError("Data validation failed.")
        
    logger.info("Data validation passed successfully.")
    return df

def prepare_for_db_insertion(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Converts DataFrame to list of dictionaries suitable for SQLAlchemy bulk insertion.
    Cleans up column names if necessary (e.g., lowercasing).
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Map column names to DB model fields
    column_mapping = {
        'CustomerId': 'customer_id',
        'Surname': 'surname',
        'CreditScore': 'credit_score',
        'Geography': 'geography',
        'Gender': 'gender',
        'Age': 'age',
        'Tenure': 'tenure',
        'Balance': 'balance',
        'NumOfProducts': 'num_of_products',
        'HasCrCard': 'has_cr_card',
        'IsActiveMember': 'is_active_member',
        'EstimatedSalary': 'estimated_salary',
        'Exited': 'exited'
    }
    
    # Drop RowNumber as it's not needed in the DB
    if 'RowNumber' in df_clean.columns:
        df_clean = df_clean.drop('RowNumber', axis=1)
        
    df_clean = df_clean.rename(columns=column_mapping)
    
    return df_clean.to_dict(orient="records")
