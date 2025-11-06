from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import logging
from typing import Dict
import numpy as np
logger = logging.getLogger(__name__)


def train_prediction_model(df: pd.DataFrame, target_col: str = 'cal_burned') -> LinearRegression:
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
    return {'cal_burned': prediction}
def load_items(path: str = 'data/items.csv') -> pd.DataFrame:
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
    return df


def recommend_plans(user_profile: np.ndarray, items_df: pd.DataFrame, top_n: int = 3):
    """Recommend plans based on similarity between user profile and item features."""
    # Giả sử ta dùng similarity = 1 / (1 + |diff|)
    items_df['score'] = -abs(items_df['difficulty'] - user_profile[0]) \
                        - abs(items_df['duration_min'] / 60 - user_profile[1]) \
                        + np.random.random(len(items_df)) * 0.1
    top_items = items_df.sort_values('score', ascending=False).head(top_n)
    logger.info(f"Top {top_n} recommended plans generated.")
    return top_items[['plan_id', 'name', 'focus']]