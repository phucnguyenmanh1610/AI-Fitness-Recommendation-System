import os
import pytest
import pandas as pd
from src.prediction.model import train_prediction_model, load_model, predict_health, MODEL_PATH, TARGET_COLS


@pytest.fixture(scope="module")
def prepared_df():
    merged_csv = r"C:\Users\Phuc\PycharmProjects\AI-Fitness-Recommendation-System\data\processed\merged.csv"
    df_processed = pd.read_csv(merged_csv)
    return df_processed


def test_train_model(prepared_df):
    """
    Kiểm tra train model và lưu model
    """
    df = prepared_df
    model, mappings = train_prediction_model(df)


    assert os.path.exists(MODEL_PATH)
    assert model is not None


    model_loaded, loaded_mappings = load_model()
    assert model_loaded is not None
    assert loaded_mappings is not None


def test_predict_health(prepared_df):
    """
    Kiểm tra predict health cho nhiều dòng và nhiều chỉ số
    """
    df = prepared_df
    model, mappings = train_prediction_model(df)


    sample_df = df.drop(columns=TARGET_COLS).sample(10, random_state=42)

    print("=== Sample predictions ===")
    for i, row in sample_df.iterrows():
        features = row.to_dict()
        pred = predict_health(model, features, mappings)
        print(f"Sample {i}: Features: {features}")
        print(f"Sample {i}: Prediction: {pred}")


        for col in TARGET_COLS:
            assert col in pred
            assert isinstance(pred[col], float)
