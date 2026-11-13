import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "chunk_metadata.json")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

if not os.path.exists(FAISS_PATH):
    print("Index not found — rebuilding from chunk_metadata.json...")
    from openai import OpenAI
    import src.retrieval.indexer as idx_module
    idx_module.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    from src.retrieval.indexer import HybridRetriever
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    retriever = HybridRetriever()
    retriever.build_index(chunks)
    print(f"✓ Index rebuilt: {len(chunks)} chunks")

from openai import OpenAI
import src.generation.rag_chain as gen_module
gen_module.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
from src.generation.rag_chain import GCCRegulatoryRAG
import gradio as gr

rag = GCCRegulatoryRAG()
rag.load()

SAMPLE_QUESTIONS = [
    "What is the minimum CET1 capital ratio required by CBUAE?",
    "What AML training obligations apply to DFSA-regulated firms?",
    "What does the IMF say about UAE financial sector vulnerabilities?",
    "What are the LCR and NSFR requirements for UAE banks?",
    "What D-SIB surcharge applies to systemically important UAE banks?",
]

def query_rag(question):
    if not question.strip():
        return "Please enter a question.", ""
    try:
        result = rag.query(question)
        answer = result["answer"]
        confidence = "HIGH"
        if "[Confidence: MEDIUM]" in answer: confidence = "MEDIUM"
        elif "[Confidence: LOW]" in answer: confidence = "LOW"
        for tag in ["[Confidence: HIGH]", "[Confidence: MEDIUM]", "[Confidence: LOW]"]:
            answer = answer.replace(tag, "").strip()
        sources_md = "\n\n".join([
            f"**[{s['rank']}]** {s['citation']} · `{s['regulator']}`"
            for s in result["sources"]
        ])
        return f"**Confidence: {confidence}**\n\n{answer}", sources_md
    except Exception as e:
        return f"Error: {str(e)}", ""

with gr.Blocks(title="GCC Regulatory RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🏦 GCC Regulatory Intelligence
**AI-powered Q&A over UAE financial regulations**
*CBUAE · DFSA · SCA · IMF — 189 chunks, hybrid retrieval (FAISS + BM25 + GraphRAG)*
""")
    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Ask a regulatory question",
                placeholder="e.g. What is the minimum CET1 ratio required by CBUAE?",
                lines=2
            )
            submit = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("**Sample questions:**")
            for q in SAMPLE_QUESTIONS:
                gr.Button(q, size="sm").click(fn=lambda x=q: x, outputs=question)
    answer_box = gr.Markdown(label="Answer")
    sources_box = gr.Markdown(label="Sources")
    submit.click(fn=query_rag, inputs=question, outputs=[answer_box, sources_box])
    question.submit(fn=query_rag, inputs=question, outputs=[answer_box, sources_box])
    gr.Markdown("---\n**RAGAs:** Faithfulness 0.926 · Answer Relevancy 0.961 | Built by Joseph")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)