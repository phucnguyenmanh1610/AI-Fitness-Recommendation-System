from .input import load_data
from .preprocess import preprocess_data

def get_processed_data(csv_path=None):
    """
    Load và preprocess data.
    Nếu csv_path không có thì dùng default hoặc synthetic data.
    """
    df = load_data(csv_path)
    df_processed = preprocess_data(df)
    return df_processed
