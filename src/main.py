import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import logging
from src.data_input.input import load_data, get_synthetic_data
from src.data_input.preprocess import preprocess_data
from src.prediction.model import train_prediction_model, predict_health
from src.recommendation.recommender import load_items, recommend_plans
from src.output.dashboard import display_dashboard
import numpy as np

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Step 1: Load real or synthetic data
    try:
        df_raw = load_data('data/raw/fitness.csv')  # Real Kaggle data
    except FileNotFoundError:
        df_raw = get_synthetic_data(500)  # Fallback to synthetic

    df_processed = preprocess_data(df_raw)

    # Add target if missing
    if 'cal_burned' not in df_processed.columns:
        df_processed['cal_burned'] = df_processed['daily_steps'] * 0.04  # Fake if needed

    # Step 2: Train & Predict
    model = train_prediction_model(df_processed, target_col='cal_burned')
    sample_features = df_processed.drop('cal_burned', axis=1).iloc[0].to_dict()
    predictions = predict_health(model, sample_features)

    # Step 3: Recommend
    user_profile = np.array(list(sample_features.values())[:3])  # Adjust dim
    items_df = load_items()
    recommendations = recommend_plans(user_profile, items_df)

    # Step 4: Output
    display_dashboard(predictions, recommendations)