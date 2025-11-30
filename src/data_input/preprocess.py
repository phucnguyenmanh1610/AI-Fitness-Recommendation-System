import pandas as pd
from .normalize import standardize_values

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final preprocessing before ML:
    - Map categorical to numeric
    - Fill NaN
    - Convert numeric types
    """
    df = df.copy()
    df = standardize_values(df)
    if "water_intake" not in df.columns:
        df["water_intake"] = df["weight"] * 0.03  # công thức hợp lý: 30ml/kg

    # --- Map categorical to numeric (overwrite numeric columns) ---
    cat_mappings = {
        "gender": {"Male": 0, "Female": 1},
        "workout_type": {"None": 0, "Cardio": 1, "Strength": 2, "Yoga": 3},
        "experience_level": {1: 1, 2: 2, 3: 3}
    }

    for col, mapping in cat_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            df[col] = df[col].fillna(-1)  # FIXED

    # --- Fill missing numeric ---
    numeric_cols = [
        "age", "height", "weight", "bmi", "max_bpm", "avg_bpm", "resting_bpm",
        "session_duration", "calories_burned", "fat_percentage", "water_intake",
        "workout_frequency", "heart_rate", "steps", "sleep_hours"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())  # FIXED

    return df
