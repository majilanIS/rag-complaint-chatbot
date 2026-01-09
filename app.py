import gradio as gr
from src.rag_pipeline import RAGPipeline
from src.rag_pipeline import FAISS_INDEX_PATH, METADATA_PATH

# ===============================
# Initialize RAG Pipeline
# ===============================
rag = RAGPipeline(
    faiss_index_path=FAISS_INDEX_PATH,
    metadata_path=METADATA_PATH
)

# ===============================
# Chat Function
# ===============================
def answer_question(question, k=5):
    if not question.strip():
        return "Please enter a question.", ""
    
    # Get answer and sources from RAG pipeline
    answer, sources = rag.generate_answer(question, k)
    
    # Convert answer to string (if it is dict, extract main key like complaint_id)
    if isinstance(answer, dict):
        if "complaint_id" in answer:
            answer_text = f"complaint_id is {answer['complaint_id']}"
        else:
            answer_text = str(answer)
    else:
        answer_text = str(answer)
    
    # Format top 3 sources as string
    sources_text = "\n\n".join([f"• complaint_id: {chunk['complaint_id']}, product: {chunk.get('product','N/A')}" 
                                if isinstance(chunk, dict) else f"• {chunk}" 
                                for chunk, _ in sources[:3]])
    
    return answer_text, sources_text

# ===============================
# Gradio UI
# ===============================
with gr.Blocks() as demo:
    gr.Markdown("## 🏦 RAG Complaint Chatbot")
    gr.Markdown("Ask questions about customer complaints and get AI-generated answers with sources.")
    
    with gr.Row():
        with gr.Column(scale=8):
            question_input = gr.Textbox(label="Your Question", placeholder="Type your question here...", lines=2)
        with gr.Column(scale=2):
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")
    
    answer_output = gr.Textbox(label="Answer", interactive=False, lines=4)
    sources_output = gr.Textbox(label="Sources", interactive=False, lines=6)

    submit_btn.click(answer_question, inputs=question_input, outputs=[answer_output, sources_output])
    clear_btn.click(lambda: ("", ""), inputs=None, outputs=[answer_output, sources_output])
    
demo.launch()
