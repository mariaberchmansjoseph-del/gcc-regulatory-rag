import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
import src.generation.rag_chain as gen_module
gen_module.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

from src.generation.rag_chain import GCCRegulatoryRAG
from api.main import app as fastapi_app
import gradio as gr
from fastapi import FastAPI
import uvicorn

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
        for tag in ["[Confidence: HIGH]", "[Confidence: MEDIUM]", "[Confidence: LOW]"]:
            answer = answer.replace(tag, "").strip()

        confidence = "HIGH"
        if "[Confidence: MEDIUM]" in result["answer"]: confidence = "MEDIUM"
        elif "[Confidence: LOW]" in result["answer"]: confidence = "LOW"

        sources_md = "\n".join([
            f"**[{s['rank']}]** {s['citation']} · `{s['regulator']}`"
            for s in result["sources"]
        ])

        full_answer = f"**Confidence: {confidence}**\n\n{answer}"
        return full_answer, sources_md
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
                gr.Button(q, size="sm").click(
                    fn=lambda x=q: x, outputs=question
                )

    with gr.Row():
        answer_box = gr.Markdown(label="Answer")

    with gr.Row():
        sources_box = gr.Markdown(label="Sources")

    submit.click(fn=query_rag, inputs=question, outputs=[answer_box, sources_box])
    question.submit(fn=query_rag, inputs=question, outputs=[answer_box, sources_box])

    gr.Markdown("""
---
**Evaluation scores (RAGAs):** Faithfulness: 0.926 · Answer Relevancy: 0.961
Built by Joseph | [GitHub](https://github.com/mariaberchmansjoseph-del/gcc-regulatory-rag)
""")

fastapi_app.mount("/", gr.mount_gradio_app(FastAPI(), demo, path="/ui"))

if __name__ == "__main__":
    import threading, time
    
    def run_fastapi():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
    
    t = threading.Thread(target=run_fastapi)
    t.start()
    
    demo.launch(server_port=7861, share=False)
