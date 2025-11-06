import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Bước 1: Chuẩn hóa tên cột ---
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # --- Bước 2: Đảm bảo có các cột bắt buộc ---
    required_cols = [
        'age', 'gender', 'weight', 'height', 'max_bpm', 'avg_bpm', 'resting_bpm',
        'session_duration', 'cal_burned', 'workout_type', 'fat_percentage',
        'water_intake', 'workout_frequency', 'experience_level'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # --- Bước 3: Tính toán đặc trưng phụ ---
    df['bmi'] = df.apply(
        lambda row: row['weight'] / (row['height'] ** 2)
        if pd.notnull(row['weight']) and pd.notnull(row['height']) and row['height'] > 0
        else np.nan,
        axis=1
    )

    # --- Bước 4: Mã hóa dữ liệu phân loại ---
    cat_cols = ['gender', 'workout_type', 'experience_level']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna("Unknown")
            df[col] = le.fit_transform(df[col])

    # --- Bước 5: Chuyển tất cả dữ liệu về kiểu số ---
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Bước 6: Điền giá trị thiếu ---
    df.fillna(0, inplace=True)

    return df
