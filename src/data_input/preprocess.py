import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from .input import calculate_bmr  # Import from same package

logger = logging.getLogger(__name__)


def calculate_bmi(weight: float, height: float) -> float:
    return weight / ((height / 100) ** 2)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data.
    """
    # Handle missing
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Add placeholders if missing columns
    if 'age' not in df.columns:
        df['age'] = np.random.randint(18, 65, len(df))
    # Tương tự cho other columns if needed

    # Calculate features
    df['BMI'] = df.apply(lambda row: calculate_bmi(row['weight'], row['height']), axis=1)
    df['BMR'] = df.apply(lambda row: calculate_bmr(row['age'], row['gender'], row['weight'], row['height']), axis=1)

    # Encoding
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Scaling
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    logger.info(f"Preprocessed data shape: {df.shape}")
    return df