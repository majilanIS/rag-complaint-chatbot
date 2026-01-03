from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate vector embeddings for text chunks.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings

# Example usage
if __name__ == "__main__":
    from text_chunking import split_texts
    from load_data import load_cleaned_data
    from sample_data import stratified_sample

    df = load_cleaned_data()
    sample_df = stratified_sample(df)
    chunks, metadata = split_texts(sample_df)
    embeddings = generate_embeddings(chunks)
