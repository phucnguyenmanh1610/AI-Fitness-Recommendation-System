import os
import logging
from typing import Dict, List, Tuple

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)
MODEL_PATH = "models/prediction_model.pkl"
MAPPINGS_PATH = "models/categorical_mappings.pkl"

# Danh sách cột categorical cần encode
CATEGORICAL_COLS = ["gender", "activity_level", "workout_type"]

# Danh sách các target cần dự đoán
TARGET_COLS = ["calories_burned", "bmi", "heart_rate", "sleep_hours", "water_intake"]


def encode_categoricals(df: pd.DataFrame, mappings: Dict[str, Dict] = None) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode các cột categorical thành số.
    Nếu mappings có sẵn, dùng mappings đó để encode.
    Trả về dataframe đã encode và mappings mới (hoặc mappings cũ).
    """
    df_encoded = df.copy()
    new_mappings = {} if mappings is None else mappings.copy()

    for col in CATEGORICAL_COLS:
        if col in df_encoded.columns:
            if mappings and col in mappings:
                # Dùng mapping có sẵn
                df_encoded[col] = df_encoded[col].map(mappings[col])
            else:
                # Tạo mapping mới
                df_encoded[col] = df_encoded[col].astype("category")
                new_mappings[col] = dict(enumerate(df_encoded[col].cat.categories))
                inv_map = {v: k for k, v in new_mappings[col].items()}
                df_encoded[col] = df_encoded[col].map(inv_map)

    return df_encoded, new_mappings


def train_prediction_model(df: pd.DataFrame,
                           target_cols: List[str] = TARGET_COLS,
                           model_path: str = MODEL_PATH,
                           mappings_path: str = MAPPINGS_PATH) -> Tuple[MultiOutputRegressor, Dict]:
    """
    Train MultiOutput Linear Regression model để dự đoán nhiều target.
    Encode categorical columns tự động.
    Lưu model và mappings.
    """
    # Kiểm tra target có tồn tại
    if not all(col in df.columns for col in target_cols):
        missing = [col for col in target_cols if col not in df.columns]
        raise ValueError(f"Target columns missing: {missing}")

    # Encode categorical
    df_encoded, mappings = encode_categoricals(df)

    X = df_encoded.drop(columns=target_cols)
    y = df_encoded[target_cols]

    if len(df) <= 1:
        logger.warning("Dataset too small, training without evaluation")
        model = MultiOutputRegressor(LinearRegression()).fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # Log MSE từng target
        for i, col in enumerate(target_cols):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            logger.info(f"MSE {col}: {mse:.2f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(mappings, mappings_path)
    logger.info(f"Model saved at {model_path}")
    logger.info(f"Categorical mappings saved at {mappings_path}")

    return model, mappings


def load_model(model_path: str = MODEL_PATH, mappings_path: str = MAPPINGS_PATH) -> Tuple[MultiOutputRegressor, Dict]:
    """
    Load model và mappings
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Categorical mappings file not found at {mappings_path}")

    model = joblib.load(model_path)
    mappings = joblib.load(mappings_path)
    return model, mappings


def predict_health(model: MultiOutputRegressor, features: Dict[str, float], mappings: Dict[str, Dict] = None) -> Dict[str, float]:
    """
    Dự đoán nhiều chỉ số sức khỏe cùng lúc.
    Tự động encode categorical features nếu mappings có sẵn.
    """
    # Encode categorical input
    if mappings:
        for col in CATEGORICAL_COLS:
            if col in features and col in mappings:
                inv_map = {v: k for k, v in mappings[col].items()}
                features[col] = inv_map[features[col]]

    df_input = pd.DataFrame([features])
    preds = model.predict(df_input)[0]

    return dict(zip(TARGET_COLS, preds))
