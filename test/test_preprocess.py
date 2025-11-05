import pytest
from src.data_input.preprocess import preprocess_data
import pandas as pd

def test_preprocess_data():
    df = pd.DataFrame({'age': [30], 'height': [175], 'weight': [70]})
    processed = preprocess_data(df)
    assert 'BMI' in processed.columns
    assert processed['BMI'].iloc[0] == pytest.approx(22.86, 0.01)  # Expected BMI