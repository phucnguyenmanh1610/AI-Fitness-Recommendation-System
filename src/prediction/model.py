import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict
import joblib
import logging

logger = logging.getLogger(__name__)

def train_prediction_model(df: pd.DataFrame, target_col: str = 'cal_burned') -> LinearRegression:
    x = df.drop(target_col, axis=1, errors='ignore')
    y = df[target_col] if target_col in df else pd.Series([0] * len(df))

    model = LinearRegression()

    if len(df) <= 1:
        model.fit(x, y)
        logger.info("Trained on small data (no evaluation)")
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Model trained: MSE={mse:.2f}, R2={r2:.2f}")

    # 🧩 Đảm bảo thư mục models tồn tại
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/prediction_model.pkl')
    return model


def predict_health(model: LinearRegression, features: Dict[str, float]) -> Dict[str, float]:
    df_input = pd.DataFrame([features])
    prediction = model.predict(df_input)[0]
    return {'cal_burned': prediction}
