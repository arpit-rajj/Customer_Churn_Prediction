import pandas as pd
from src.features.engineering import FeatureEngineer

def test_feature_engineering():
    df = pd.DataFrame({
        'Balance': [1000, 0, 500],
        'NumOfProducts': [2, 1, 0],
        'Tenure': [5, 0, 10],
        'Age': [30, 40, 50],
        'IsActiveMember': [1, 0, 1]
    })
    
    engineer = FeatureEngineer()
    df_eng = engineer.transform(df)
    
    assert 'BalancePerProduct' in df_eng.columns
    assert 'IsZeroBalance' in df_eng.columns
    assert 'ProductsPerTenure' in df_eng.columns
    assert 'ActiveByAge' in df_eng.columns
    
    assert df_eng['BalancePerProduct'].iloc[0] == 500.0
    assert df_eng['IsZeroBalance'].iloc[1] == 1
    assert df_eng['ActiveByAge'].iloc[1] == 0
    assert df_eng['ActiveByAge'].iloc[2] == 50
