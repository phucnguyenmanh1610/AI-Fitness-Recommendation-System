from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)

def train_prediction_model(df: pd.DataFrame, target_col: str = 'calories_burned') -> LinearRegression:
    x = df.drop(target_col, axis=1, errors='ignore')
    y = df[target_col] if target_col in df else pd.Series([0] * len(df))
    if len(df) <= 1:
        model = LinearRegression()
        model.fit(x, y)
        logger.info("Trained on small data")
        joblib.dump(model, 'models/prediction_model.pkl')
        return model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Model trained: MSE={mse:.2f}, R2={r2:.2f}")
    joblib.dump(model, 'models/prediction_model.pkl')
    return model

def predict_health(model: LinearRegression, features: Dict[str, float]) -> Dict[str, float]:
    df_input = pd.DataFrame([features])
    prediction = model.predict(df_input)[0]
    return {'calories_burned': prediction}

def load_items(path: str = 'data/train/items.csv') -> pd.DataFrame:
    """Load available fitness or diet plans."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} items from {path}")
    except FileNotFoundError:
        # Fallback synthetic items
        df = pd.DataFrame({
            'plan_id': range(5),
            'name': ['Yoga', 'Cardio', 'Strength', 'Pilates', 'HIIT'],
            'difficulty': np.random.randint(1, 5, 5),
            'duration_min': np.random.randint(20, 60, 5),
            'focus': ['flexibility', 'endurance', 'strength', 'core', 'full-body']
        })
        logger.warning("Items file not found, using synthetic data.")

    # Đảm bảo có cột plan_id (nếu không thì tạo mới)
    if 'plan_id' not in df.columns:
        if 'id' in df.columns:
            df.rename(columns={'id': 'plan_id'}, inplace=True)
            logger.info("Renamed 'id' column to 'plan_id'")
        else:
            df['plan_id'] = range(1, len(df) + 1)
            logger.warning("plan_id column not found, auto-generated sequential IDs")

    return df

def compute_score(user_profile, difficulty, duration_min):
    """
    Compute a recommendation score based on user's difficulty preference and workout duration preference.
    """
    pref_diff = user_profile[0]
    pref_dur = user_profile[1]
    dur_scaled = duration_min / 60.0  # normalize duration to 0-1
    diff_score = max(0, 1 - abs(difficulty - pref_diff) / 4)  # normalize difference score
    dur_score = max(0, 1 - abs(dur_scaled - pref_dur))
    score = 0.6 * diff_score + 0.4 * dur_score
    return score

def recommend_plans(user_profile, items_df, top_n=3, include_score=False):
    """
    Recommend top N plans from items_df based on user_profile preferences.

    Args:
      user_profile: list or tuple [difficulty_preference (1-5), duration_scaled_0_1]
      items_df: DataFrame containing plans
      top_n: Number of recommendations to return
      include_score: Whether to include score column in output

    Returns:
      DataFrame with recommended plans (columns: plan_id, name, focus[, score])
    """
    items_df = items_df.copy()

    # Đảm bảo tồn tại cột plan_id trước khi sort
    if 'plan_id' not in items_df.columns:
        if 'id' in items_df.columns:
            items_df.rename(columns={'id': 'plan_id'}, inplace=True)
            logger.info("Renamed 'id' column to 'plan_id' in recommend_plans")
        else:
            items_df['plan_id'] = range(1, len(items_df) + 1)
            logger.warning("plan_id column missing, auto-generated sequential IDs in recommend_plans")

    # Tính điểm score cho từng plan
    items_df['score'] = items_df.apply(
        lambda row: compute_score(user_profile, row['difficulty'], row['duration_min']),
        axis=1
    )


    sorted_items = items_df.sort_values(['score', 'plan_id'], ascending=[False, True])


    if top_n > len(sorted_items):
        repeats = int(np.ceil(top_n / len(sorted_items)))
        repeated_items = pd.concat([sorted_items] * repeats, ignore_index=True)
        repeated_items = repeated_items.sort_values(['score', 'plan_id'], ascending=[False, True])
        result_items = repeated_items.head(top_n)
    else:
        result_items = sorted_items.head(top_n)

    columns = ['plan_id', 'name', 'focus']
    if include_score:
        columns.append('score')

    return result_items[columns]
