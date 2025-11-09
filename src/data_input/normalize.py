import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def normalize_csv_files(raw_dir="data/raw", output_path="data/processed/merged.csv"):
    raw_dir = Path(raw_dir)
    all_files = list(raw_dir.glob("*.csv"))
    merged = []

    for f in all_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                logger.warning(f"⚠️ {f.name} is empty, skipping.")
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
    logger.info(f"✅ Merged {len(merged)} CSVs into {output_path} with shape {final_df.shape}")
    return final_df


def standardize_columns(df):
    """
    Chuẩn hóa tên cột và thêm cột trống nếu thiếu.
    """
    standard_cols = [
        "user_id", "age", "gender", "height", "weight",
        "activity_level", "goal", "duration",
        "plan_name", "calories_burned", "bmi", "heart_rate",
        "sleep_hours", "steps", "mood"
    ]

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in standard_cols:
        if col not in df.columns:
            df[col] = None

    return df[standard_cols]
