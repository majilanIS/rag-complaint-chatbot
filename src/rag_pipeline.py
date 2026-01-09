# rag_pipeline.py

import os
import pickle
import faiss
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ================================
# Configuration / Paths
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.idx")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

# ================================
# Helper function
# ================================
def chunk_to_text(chunk):
    """
    Convert a dictionary chunk into readable text.
    """
    if isinstance(chunk, dict):
        return ", ".join([f"{k}: {v}" for k, v in chunk.items()])
    return str(chunk)

# ================================
# RAG Pipeline Class
# ================================
class RAGPipeline:
    def __init__(self, faiss_index_path, metadata_path,
                 embedding_model_name="all-MiniLM-L6-v2",
                 llm_model_name="google/flan-t5-small",
                 max_new_tokens=200):

        # Load embedding model
        self.embed_model = SentenceTransformer(embedding_model_name)

        # Load FAISS index
        self.vector_index = faiss.read_index(faiss_index_path)

        # Load text chunks (metadata)
        with open(metadata_path, "rb") as f:
            self.text_chunks = pickle.load(f)

        print(f"[INFO] Loaded {len(self.text_chunks)} text chunks.")

        # Initialize LLM generator
        self.generator = pipeline(
            "text2text-generation",
            model=llm_model_name,
            max_new_tokens=max_new_tokens,
            device=-1  # use CPU; set 0 for GPU if available
        )

    # ----------------------------
    # Retrieval
    # ----------------------------
    def retrieve_top_k(self, question, k=5):
        """
        Retrieve top-k most relevant text chunks with distances.
        Returns: List of (chunk, distance)
        """
        q_vec = self.embed_model.encode([question], normalize_embeddings=True)
        distances, indices = self.vector_index.search(q_vec, k)
        results = [(self.text_chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
        return results

    # ----------------------------
    # Prompt Builder
    # ----------------------------
    def build_prompt(self, question, retrieved_chunks):
        """
        Build a prompt instructing the LLM to answer using only retrieved context.
        """
        context = "\n".join([chunk_to_text(chunk) for chunk, _ in retrieved_chunks])
        prompt = f"""
You are a financial analyst assistant for CrediTrust.
Answer the question using ONLY the following context.
If the answer is not in the context, say you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""
        return prompt.strip()

    # ----------------------------
    # Generate Answer
    # ----------------------------
    def generate_answer(self, question, k=5):
        """
        Full RAG pipeline: retrieve, build prompt, generate answer.
        Returns: answer text and retrieved chunks
        """
        retrieved_chunks = self.retrieve_top_k(question, k)
        prompt = self.build_prompt(question, retrieved_chunks)
        output = self.generator(prompt)
        answer = output[0]["generated_text"].rsplit("Answer:", 1)[-1].strip()
        return answer, retrieved_chunks

    # ----------------------------
    # Evaluation Function
    # ----------------------------
    def evaluate_questions(self, questions, k=5, output_file="rag_evaluation.csv"):
        """
        Evaluate multiple questions, generate a table with:
        Question | Generated Answer | Top Sources | Quality Score | Comments
        """
        evaluation = []
        for q in questions:
            answer, retrieved = self.generate_answer(q, k)
            top_sources = [chunk_to_text(chunk) for chunk, _ in retrieved[:2]]
            evaluation.append({
                "Question": q,
                "Generated Answer": answer,
                "Retrieved Sources": " | ".join(top_sources),
                "Quality Score": "",
                "Comments/Analysis": ""
            })

        df = pd.DataFrame(evaluation)
        df.to_csv(output_file, index=False)
        print(f"[INFO] Evaluation table saved to {output_file}")
        return df

# ================================
# Test block
# ================================
if __name__ == "__main__":
    rag = RAGPipeline(FAISS_INDEX_PATH, METADATA_PATH)

    sample_questions = [
        "What are common reasons for customer complaints?",
        "Which complaint category receives the highest severity score?",
        "How many customers reported delayed payments?",
        "What actions were taken to resolve complaints?",
        "Are there trends in complaints over the past year?"
    ]

    for q in sample_questions:
        print("\n" + "="*80)
        print(f"[Question]: {q}")
        answer, sources = rag.generate_answer(q)
        print(f"[Generated Answer]: {answer}\n")
        print("[Top 2 Sources]:")
        for chunk, dist in sources[:2]:
            print(f"- {chunk_to_text(chunk)[:200]}... (distance: {dist:.4f})")

    # Generate evaluation table automatically
    df_eval = rag.evaluate_questions(sample_questions)
    print("\n[Evaluation Table Preview]")
    print(df_eval.head())

