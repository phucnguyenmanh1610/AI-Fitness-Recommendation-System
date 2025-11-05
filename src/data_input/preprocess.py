import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def calculate_bmi(weight: float, height: float) -> float:
    """Calculate BMI: weight (kg) / (height (m)^2)"""
    return weight / ((height / 100) ** 2)


def calculate_bmr(age: int, gender: str, weight: float, height: float) -> float:
    """BMR formula: 10*W + 6.25*H - 5*A + s (s=5 male, -161 female)"""
    s = 5 if gender == 'male' else -161
    return 10 * weight + 6.25 * height - 5 * age + s


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess: Handle missing, calculate features, scale.
    :param df: Raw DataFrame
    :return: Processed DataFrame (scaled [0,1])
    """
    # Handle missing (mean substitution)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Calculate additional features
    df['BMI'] = df.apply(lambda row: calculate_bmi(row['weight'], row['height']), axis=1)
    df['BMR'] = df.apply(lambda row: calculate_bmr(row['age'], row['gender'], row['weight'], row['height']), axis=1)

    # Scaling (only numeric columns)
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    logger.info(f"Preprocessed data shape: {df.shape}")
    return df