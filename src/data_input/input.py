import pandas as pd
import numpy as np
import logging
from src.data_input.normalize import normalize_csv_files


logger = logging.getLogger(__name__)
def load_data(file_path=None):
    """
    Load and normalize dataset.
    If file_path is provided, try to load that file; otherwise normalize all CSVs.
    """
    if file_path:
        import pandas as pd
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df

    df = normalize_csv_files()
    if df is None:
        raise ValueError("No valid data to train on.")
    return df


def get_synthetic_data(n_samples: int = 100):
    """Generate synthetic fitness data for testing or fallback."""
    logging.warning("Generating synthetic data instead of loading from file.")
    np.random.seed(42)
    df = pd.DataFrame({
        "Age": np.random.randint(18, 60, n_samples),
        "Gender": np.random.choice(["Male", "Female"], n_samples),
        "Weight (kg)": np.random.uniform(45, 100, n_samples),
        "Height (m)": np.random.uniform(1.5, 2.0, n_samples),
        "Max_BPM": np.random.randint(120, 200, n_samples),
        "Avg_BPM": np.random.randint(90, 160, n_samples),
        "Resting_BPM": np.random.randint(60, 90, n_samples),
        "Session_Duration (hours)": np.random.uniform(0.5, 2.0, n_samples),
        "Calories_Burned": np.random.uniform(200, 900, n_samples),
        "Workout_Type": np.random.choice(["Cardio", "Strength", "Yoga"], n_samples),
        "Fat_Percentage": np.random.uniform(10, 30, n_samples),
        "Water_Intake (liters)": np.random.uniform(1.0, 3.0, n_samples),
        "Workout_Frequency (days/week)": np.random.randint(1, 7, n_samples),
        "Experience_Level": np.random.randint(1, 3, n_samples),
        "BMI": np.random.uniform(18.0, 35.0, n_samples),
    })
    return df
