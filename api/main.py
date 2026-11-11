from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os

sys.path.insert(0, "/content/gcc-regulatory-rag")
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

from openai import OpenAI
import src.generation.rag_chain as gen_module
gen_module.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

from src.generation.rag_chain import GCCRegulatoryRAG

app = FastAPI(title="GCC Regulatory RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = GCCRegulatoryRAG()
rag.load()

class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    citation: str
    regulator: str
    rank: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: str
    graph_augmented: bool

@app.get("/")
def root():
    return {"status": "ok", "service": "GCC Regulatory RAG", "chunks": len(rag.retriever.chunks)}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = rag.query(req.question)
        confidence = "HIGH"
        if "[Confidence: MEDIUM]" in result["answer"]:
            confidence = "MEDIUM"
        elif "[Confidence: LOW]" in result["answer"]:
            confidence = "LOW"
        answer = result["answer"].replace("[Confidence: HIGH]","").replace("[Confidence: MEDIUM]","").replace("[Confidence: LOW]","").strip()
        return QueryResponse(
            answer=answer,
            sources=[Source(citation=s["citation"], regulator=s["regulator"], rank=s["rank"]) for s in result["sources"]],
            confidence=confidence,
            graph_augmented=any(s.get("graph_augmented") for s in result["sources"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "chunks_loaded": len(rag.retriever.chunks)}
