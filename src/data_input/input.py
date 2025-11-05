import pandas as pd
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    :param file_path: Path to CSV (e.g., 'data/raw/fitness.csv')
    :return: Pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

def get_user_input() -> Dict[str, float]:
    """
    Simulate user input from wearable or manual entry.
    TODO: Integrate with API (e.g., Fitbit) later.
    :return: Dict of features
    """
    # Placeholder: Hardcoded for prototype
    return {
        'age': 30,
        'gender': 'male',  # 'male' or 'female'
        'height': 175.0,   # cm
        'weight': 70.0,    # kg
        'daily_steps': 8000,
        'heart_rate': 75,
        'sleep_time': 7.5, # hours
        'calorie_intake': 2500.0
    }