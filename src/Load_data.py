# src/load_data.py
import pandas as pd

def load_cleaned_data(file_path="../data/filtered_complaints.csv"):
    """
    Load the cleaned complaint dataset.
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} complaints from {file_path}")
    return df

# Example usage
if __name__ == "__main__":
    df = load_cleaned_data()
    print(df.head())
