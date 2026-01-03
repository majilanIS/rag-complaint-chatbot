import pandas as pd

def stratified_sample(df, sample_size=12000):
    """
    Create a stratified sample across product categories.
    """
    stratified_sample = df.groupby("Product", group_keys=False).apply(
        lambda x: x.sample(int(sample_size * len(x) / len(df)), random_state=42)
    )
    print(f"Sampled {len(stratified_sample)} complaints (stratified by product)")
    return stratified_sample

# Example usage
if __name__ == "__main__":
    from load_data import load_cleaned_data
    df = load_cleaned_data()
    sample_df = stratified_sample(df)
    print(sample_df['Product'].value_counts())
