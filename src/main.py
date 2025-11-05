import logging
from src.data_input.input import load_data, get_user_input
from src.data_input.preprocess import preprocess_data
from src.prediction.model import train_prediction_model, predict_health
from src.recommendation.recommender import load_items, recommend_plans
from src.output.dashboard import display_dashboard
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Step 1: Load & Preprocess
    # df_raw = load_data('data/raw/fitness.csv')  # Uncomment when have data
    user_features = get_user_input()  # Placeholder
    df_processed = preprocess_data(pd.DataFrame([user_features]))

    # Step 2: Train & Predict (use pre-trained if available)
    model = train_prediction_model(df_processed, target_col='cal_burned')  # Adjust target
    predictions = predict_health(model, user_features)

    # Step 3: Recommend
    user_profile = np.array(list(user_features.values()))  # Convert to vector; adjust
    items_df = load_items()
    recommendations = recommend_plans(user_profile, items_df)

    # Step 4: Output
    display_dashboard(predictions, recommendations)