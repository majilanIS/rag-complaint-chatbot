# rag-complaint-chatbot

---

## **_ task-1:EDA and Preprocessing data _**

The exploratory data analysis revealed significant variation in complaint volume across financial products, with some products dominating the dataset. A substantial portion of complaints lacked a consumer narrative, making them unsuitable for semantic retrieval tasks. Narrative length analysis showed a wide distribution, including very short and extremely long complaints, highlighting the need for preprocessing before embedding generation.

To align with project requirements, the dataset was filtered to include only complaints related to Credit cards, Personal loans, Savings accounts, and Money transfers. Records without consumer complaint narratives were removed. Text preprocessing steps such as lowercasing, removal of boilerplate language, special characters, and excessive whitespace were applied to improve embedding quality.

## The resulting cleaned dataset provides high-quality textual inputs suitable for a Retrieval-Augmented Generation (RAG) pipeline, ensuring more accurate semantic search and response generation.

---

---

## ** Task 2: Text Chunking, Embedding, and Vector Store Indexing **

**Objective:** Convert cleaned complaint narratives into vectors for efficient semantic search.

**Steps:**

1. **Stratified Sampling:** Select 10,000–15,000 complaints proportionally across product categories.
2. **Text Chunking:** Split long complaints into smaller overlapping chunks (~500 chars, 50 overlap) using a custom splitter.
3. **Embedding Generation:** Use `sentence-transformers/all-MiniLM-L6-v2` to convert chunks into vector embeddings.
4. **Vector Store Indexing:** Store embeddings in a FAISS index with metadata (complaint ID, product) for fast semantic search.

**Outcome:** Complaints are now searchable in vector space, enabling similarity queries and analysis.

## **Deliverables:** Python script, FAISS index (`vector_store/faiss_index.idx`), and metadata (`vector_store/metadata.pkl`).

---
