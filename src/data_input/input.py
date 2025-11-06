import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        # Rename columns to match (adjust based on Kaggle CSV)
        df.rename(columns={
            'StepTotal': 'daily_steps',
            'Calories': 'cal_burned',
            # Add more if needed
        }, inplace=True)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

def calculate_bmr(age: int, gender: str, weight: float, height: float) -> float:
    s = 5 if gender == 'male' else -161
    return 10 * weight + 6.25 * height - 5 * age + s

def get_activity_factor(daily_steps: int) -> float:
    if daily_steps < 5000:
        return 1.2
    elif 5000 <= daily_steps <= 10000:
        return 1.55
    else:
        return 1.725

def get_synthetic_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate synthetic data.
    """
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 65, n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples),
        'height': np.random.uniform(150, 190, n_samples),
        'weight': np.random.uniform(50, 110, n_samples),
        'daily_steps': np.random.randint(1000, 20000, n_samples),
        'heart_rate': np.random.randint(60, 100, n_samples),
        'sleep_time': np.random.uniform(4, 10, n_samples),
        'calorie_intake': np.random.uniform(1500, 3500, n_samples),
    }
    df = pd.DataFrame(data)
    df['BMR'] = df.apply(lambda row: calculate_bmr(row['age'], row['gender'], row['weight'], row['height']), axis=1)
    df['activity_factor'] = df['daily_steps'].apply(get_activity_factor)
    df['cal_burned'] = df['BMR'] * df['activity_factor'] + np.random.normal(0, 50, n_samples)
    logger.info(f"Generated synthetic data with shape {df.shape}")
    return df