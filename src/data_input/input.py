# src/data_input/input.py
import pandas as pd
import logging
from pathlib import Path
from .normalize import normalize_csv_files, standardize_values

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load a CSV file or merge all CSVs in data/raw
    """
    if file_path:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
    else:
        df = normalize_csv_files()
        if df is None:
            raise ValueError("No valid CSV files to load.")
    df = standardize_values(df)
    return df


def get_synthetic_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate synthetic fitness data
    """
    import numpy as np
    logger.warning("Generating synthetic data instead of loading from file.")
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 60, n_samples),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "height": np.round(np.random.uniform(1.5, 2.0, n_samples), 2),
        "weight": np.round(np.random.uniform(45, 120, n_samples), 1),
        "bmi": np.round(np.random.uniform(18, 35, n_samples), 2),
        "workout_type": np.random.choice(["None", "Cardio", "Strength", "Yoga"], n_samples),
        "experience_level": np.random.randint(1, 4, n_samples),
        "resting_bpm": np.random.randint(55, 85, n_samples),
        "avg_bpm": np.random.randint(75, 160, n_samples),
        "max_bpm": np.random.randint(120, 200, n_samples),
        "session_duration": np.round(np.random.uniform(0.5, 2.0, n_samples), 2),
        "calories_burned": np.round(np.random.uniform(200, 900, n_samples), 1),
        "fat_percentage": np.round(np.random.uniform(10, 35, n_samples), 1),
        "water_intake": np.round(np.random.uniform(1.0, 3.5, n_samples), 2),
        "workout_frequency": np.random.randint(1, 7, n_samples),
        "heart_rate": np.random.randint(60, 180, n_samples),
        "steps": np.random.randint(1000, 15000, n_samples),
        "sleep_hours": np.round(np.random.uniform(4, 9, n_samples), 1),
    })
    df = standardize_values(df)
    return df
