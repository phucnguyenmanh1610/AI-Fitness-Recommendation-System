import pandas as pd
from src.data_input.preprocess import preprocess_data
from tabulate import tabulate  # pip install tabulate

def test_preprocess_data_csv():
    # Read original CSV
    df = pd.read_csv(r"C:\Users\Phuc\PycharmProjects\module1\data\raw\fitness.csv")

    # Print first 5 rows of original data
    print("\nOriginal data (first 5 rows):")
    print(tabulate(df.head(), headers='keys', tablefmt='fancy_grid'))

    # Process the data
    processed = preprocess_data(df)

    # Print first 5 rows of processed data
    print("\nProcessed data (first 5 rows):")
    print(tabulate(processed.head(), headers='keys', tablefmt='fancy_grid'))

    # Check that 'bmi' column exists
    assert 'bmi' in processed.columns

    # --- Lưu processed data ra file merged.csv ---
    processed.to_csv(r"C:\Users\Phuc\PycharmProjects\AI-Fitness-Recommendation-System\data\processed\merged.csv", index=False)
    print("\n✅ Processed data saved to merged.csv")
