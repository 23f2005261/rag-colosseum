import os, re, threading, json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

with open(KB_PATH, "r", encoding="utf-8") as f:
    PASSAGES = json.load(f)

if not isinstance(PASSAGES, list) or len(PASSAGES) != 50:
    raise RuntimeError("knowledge_base.json must contain exactly 50 passages")

model = None
passage_vectors = None
load_error = None
ready = threading.Event()

app = FastAPI(title="RAG Colosseum", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: List[int]
    scores: List[float]
    confidence: float

def load_model_in_background():
    global model, passage_vectors, load_error
    try:
        from sentence_transformers import SentenceTransformer
        loaded_model = SentenceTransformer(MODEL_NAME)
        vectors = loaded_model.encode(
            PASSAGES, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")
        model = loaded_model
        passage_vectors = vectors
        print(f"RAG model ready; indexed {len(PASSAGES)} passages.", flush=True)
    except Exception as exc:
        load_error = str(exc)
        print(f"RAG model loading failed: {exc}", flush=True)
    finally:
        ready.set()

@app.on_event("startup")
def start_background_model_loading():
    threading.Thread(target=load_model_in_background, daemon=True).start()

def tokenize(s: str):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def extract_answer(question: str, passages: List[str]) -> Optional[str]:
    q_terms = tokenize(question)
    scored = []
    for passage in passages:
        for sent in re.split(r"(?<=[.!?])\s+", passage.strip()):
            terms = tokenize(sent)
            overlap = len(q_terms & terms)
            if overlap:
                scored.append((overlap / max(1, len(q_terms)), sent.strip()))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    chosen, seen = [], set()
    for _, sent in scored:
        key = sent.lower()
        if key not in seen:
            chosen.append(sent)
            seen.add(key)
        if len(chosen) == 3:
            break
    return " ".join(chosen) if chosen else None

@app.get("/")
def root():
    return {"service": "RAG Colosseum", "status": "ready" if ready.is_set() and load_error is None else "initializing",
            "endpoint": "POST /query"}

@app.get("/health")
def health():
    if not ready.is_set():
        raise HTTPException(status_code=503, detail="Embedding model is still loading")
    if load_error:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {load_error}")
    return {"status": "ok", "passages": len(PASSAGES), "embedding_model": MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not ready.is_set() or model is None or passage_vectors is None:
        raise HTTPException(status_code=503, detail="RAG model is still loading; retry shortly")
    if load_error:
        raise HTTPException(status_code=503, detail=f"RAG model unavailable: {load_error}")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    qv = model.encode([question], normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=False).astype("float32")[0]
    sims = passage_vectors @ qv
    order = np.argsort(-sims)
    best = float(sims[order[0]])

    if best < 0.27:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.15)
    if best < 0.31:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.25)

    k = min(req.top_k, len(PASSAGES))
    selected = [int(i) for i in order[:k]]
    selected_scores = [round(max(0.0, min(1.0, float(sims[i]))), 4) for i in selected]
    answer = extract_answer(question, [PASSAGES[i] for i in selected])
    if answer is None:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.2)

    confidence = max(0.61, min(0.96, 0.61 + (best - 0.31) * 0.8))
    return QueryResponse(answer=answer, sources=selected, scores=selected_scores,
                         confidence=round(confidence, 3))
