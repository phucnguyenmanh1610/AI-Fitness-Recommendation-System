import os
import logging
from typing import Dict, List

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)
MODEL_PATH = "models/prediction_model.pkl"

CATEGORICAL_COLS = ["gender", "activity_level", "workout_type"]

TARGET_COLS = ["calories_burned", "bmi", "heart_rate", "sleep_hours", "water_intake"]


# ---------------------------------------------------------
# ENCODER
# ---------------------------------------------------------
def encode_categoricals(df: pd.DataFrame, mappings=None):
    df = df.copy()
    new_map = {} if mappings is None else mappings.copy()

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        if mappings and col in mappings:
            inv = {v: k for k, v in mappings[col].items()}
            df[col] = df[col].map(inv).fillna(0)
        else:
            df[col] = df[col].astype("category")
            categories = list(df[col].cat.categories)
            new_map[col] = {i: cat for i, cat in enumerate(categories)}
            inv = {cat: i for i, cat in enumerate(categories)}
            df[col] = df[col].map(inv)

    return df, new_map


# ---------------------------------------------------------
# TRAIN MODEL – ALWAYS SAVE FULL DICT
# ---------------------------------------------------------
def train_prediction_model(df: pd.DataFrame,
                           target_cols: List[str] = TARGET_COLS,
                           model_path: str = MODEL_PATH):

    if not all(t in df.columns for t in target_cols):
        missing = [t for t in target_cols if t not in df.columns]
        raise ValueError(f"Missing target columns: {missing}")

    df_enc, mappings = encode_categoricals(df)

    X = df_enc.drop(columns=target_cols)
    y = df_enc[target_cols]

    if len(df) <= 1:
        model = MultiOutputRegressor(RandomForestRegressor()).fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = MultiOutputRegressor(RandomForestRegressor()).fit(X_train, y_train)

        pred = model.predict(X_test)
        for i, col in enumerate(target_cols):
            mse = mean_squared_error(y_test.iloc[:, i], pred[:, i])
            logger.info(f"MSE {col}: {mse:.4f}")

    # ALWAYS SAVE AS DICT (model_data)
    model_data = {
        "model": model,
        "feature_names": list(X.columns),
        "mappings": mappings
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)

    return model_data  # IMPORTANT: return dict, not tuple


# ---------------------------------------------------------
# PREDICT – USE model_data DICT
# ---------------------------------------------------------
def predict_health(model_data, features: Dict[str, float]):
    model = model_data["model"]
    mappings = model_data["mappings"]
    required_cols = model_data["feature_names"]

    feat = features.copy()

    # categorical encoding
    for col in CATEGORICAL_COLS:
        if col in feat and col in mappings:
            inv = {v: k for k, v in mappings[col].items()}
            feat[col] = inv.get(feat[col], 0)

    df_in = pd.DataFrame([feat])

    # ensure all required features exist
    for c in required_cols:
        if c not in df_in.columns:
            df_in[c] = 0

    df_in = df_in[required_cols]

    preds = model.predict(df_in)[0]

    # prevent negative values
    preds = [max(0, p) for p in preds]

    return dict(zip(TARGET_COLS, preds))
