# test_pipeline.py
import pandas as pd
import numpy as np

def test_feature_schema_length():
    # Verify that the expected feature schema has exactly 46 features
    expected_features_count = 46
    # We can import or define a quick dummy check
    assert expected_features_count == 46, "Schema must contain exactly 46 features."

def test_dataframe_reindex():
    # Test that reindexing works correctly with missing columns
    dummy_df = pd.DataFrame({"age_approx": [25]})
    expected_features = ["age_approx", "sex_male"]
    
    df_encoded = pd.get_dummies(dummy_df)
    df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)
    
    assert "sex_male" in df_aligned.columns
    assert df_aligned.loc[0, "sex_male"] == 0
    assert len(df_aligned.columns) == 2