import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import logging
from src.data_input.input import load_data, get_synthetic_data
from src.data_input.preprocess import preprocess_data
from src.prediction.model import train_prediction_model, predict_health

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # --- Step 1: Load real CSV or fallback to synthetic ---
    try:
        df_raw = load_data(r"data/raw/fitness.csv")
    except FileNotFoundError:
        logging.warning("CSV not found, using synthetic data.")
        df_raw = get_synthetic_data(500)

    # --- Step 2: Preprocess data ---
    df_processed = preprocess_data(df_raw)

    # --- Step 3: Ensure target column exists ---
    if 'cal_burned' not in df_processed.columns:
        df_processed['cal_burned'] = df_processed.get('daily_steps', 0) * 0.04

    # --- Step 4: Train model ---
    model = train_prediction_model(df_processed, target_col='cal_burned')

    # --- Step 5: Predict on a sample ---
    sample_features = df_processed.drop('cal_burned', axis=1).iloc[0].to_dict()
    predictions = predict_health(model, sample_features)

    # --- Step 6: Print results ---
    print("Sample prediction for calories burned:")
    print(predictions)
