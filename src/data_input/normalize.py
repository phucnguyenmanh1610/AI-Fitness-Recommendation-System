# src/data_input/normalize.py
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_COLS = ["user_id", "goal", "plan_name", "mood"]

def normalize_csv_files(raw_dir: str = "data/raw", output_path: str = "data/processed/merged.csv") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    all_files = list(raw_dir.glob("*.csv"))
    merged = []

    for f in all_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                logger.warning(f"{f.name} is empty, skipping.")
                continue
            df = standardize_columns(df)
            merged.append(df)
            logger.info(f"Loaded {f.name} with shape {df.shape}")
        except Exception as e:
            logger.error(f"Error reading {f.name}: {e}")

    if not merged:
        logger.warning("No valid CSV files found to merge.")
        return None

    final_df = pd.concat(merged, ignore_index=True).drop_duplicates()
    final_df.to_csv(output_path, index=False)
    logger.info(f"Merged {len(merged)} CSVs into {output_path} with shape {final_df.shape}")
    return final_df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for col in DEFAULT_COLS:
        if col not in df.columns:
            df[col] = None
    return df


def standardize_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize categorical and numeric values
    """
    df = df.copy()

    # --- Gender ---
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().replace({"M": "Male", "F": "Female"})
        df["gender_num"] = df["gender"].map({"Male": 0, "Female": 1})

    # --- Workout type ---
    if "workout_type" in df.columns:
        df["workout_type"] = df["workout_type"].astype(str).str.strip().str.capitalize()
        df["workout_type_num"] = df["workout_type"].map({"None": 0, "Cardio": 1, "Strength": 2, "Yoga": 3})

    # --- Numeric columns ---
    numeric_cols = [
        "age","height","weight","bmi","max_bpm","avg_bpm","resting_bpm",
        "session_duration","calories_burned","fat_percentage","water_intake",
        "workout_frequency","heart_rate","steps","sleep_hours"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

    return df
