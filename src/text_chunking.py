from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_texts(df, text_column="Consumer complaint narrative", chunk_size=500, chunk_overlap=50):
    """
    Split long complaints into smaller chunks.
    Returns list of chunks and associated metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    metadata = []

    for idx, row in df.iterrows():
        complaint_chunks = text_splitter.split_text(row[text_column])
        for chunk in complaint_chunks:
            chunks.append(chunk)
            metadata.append({"complaint_id": row['Complaint ID'], "product": row['Product']})

    print(f"Generated {len(chunks)} chunks from {len(df)} complaints")
    return chunks, metadata

# Example usage
if __name__ == "__main__":
    from load_data import load_cleaned_data
    from sample_data import stratified_sample

    df = load_cleaned_data()
    sample_df = stratified_sample(df)
    chunks, metadata = split_texts(sample_df)
    print(chunks[:2])
    print(metadata[:2])
