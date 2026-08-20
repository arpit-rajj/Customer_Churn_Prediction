from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.features.engineering import FeatureEngineer

def get_preprocessing_pipeline() -> Pipeline:
    """
    Returns a scikit-learn Pipeline that applies feature engineering
    and standard preprocessing (imputation, scaling, one-hot encoding).
    
    Why scaling for Random Forest?
    While Random Forests do not strictly require feature scaling because they partition data
    using decision trees, scaling is included in this shared pipeline because:
    1. We evaluate other baselines (like Logistic Regression) which DO require scaling.
    2. It does not negatively impact the Random Forest performance.
    """
    
    # Define columns to be processed
    # Note: 'CustomerId', 'Surname' should be dropped before passing to this pipeline.
    # Exited is the target.
    
    # Original numerical features + engineered features
    numeric_features = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'EstimatedSalary', 'BalancePerProduct', 'ProductsPerTenure', 'ActiveByAge'
    ]
    
    # Categorical and boolean-like features
    categorical_features = ['Geography', 'Gender']
    
    # Binary features that don't need OHE but might need imputation
    passthrough_features = ['HasCrCard', 'IsActiveMember', 'IsZeroBalance']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    passthrough_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('passthrough', passthrough_transformer, passthrough_features)
        ],
        remainder='drop' # Drop any other columns (like CustomerId, Surname if accidentally passed)
    )

    # Combine feature engineering and preprocessing
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline
