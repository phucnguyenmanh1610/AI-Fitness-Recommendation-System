from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, Dict
import logging
import joblib  # For saving models

logger = logging.getLogger(__name__)


def train_prediction_model(df: pd.DataFrame, target_col: str = 'cal_burned') -> LinearRegression:
    """
    Train regression model.
    :param df: Processed DataFrame
    :param target_col: Target to predict (e.g., 'cal_burned')
    :return: Trained model
    """
    X = df.drop(target_col, axis=1, errors='ignore')
    y = df[target_col] if target_col in df else None  # Placeholder if no target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()  # Or RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Model trained: MSE={mse:.2f}, R2={r2:.2f}")

    joblib.dump(model, 'models/prediction_model.pkl')  # Save model
    return model


def predict_health(model: LinearRegression, features: Dict[str, float]) -> Dict[str, float]:
    """
    Predict health metrics.
    :param model: Trained model
    :param features: Dict of input features
    :return: Dict of predictions {'BMI': float, 'cal_burned': float, ...}
    """
    df_input = pd.DataFrame([features])
    prediction = model.predict(df_input)[0]  # Assume single prediction
    return {'cal_burned': prediction}  # Expand as needed