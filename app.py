import os, re, math
from importlib import import_module
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

try:
    SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
except ImportError as exc:
    raise RuntimeError(
        "sentence-transformers is required; install it with "
        "'pip install sentence-transformers'"
    ) from exc

KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

with open(KB_PATH, "r", encoding="utf-8") as f:
    PASSAGES = __import__("json").load(f)

if not isinstance(PASSAGES, list) or len(PASSAGES) != 50:
    raise RuntimeError("knowledge_base.json must contain exactly 50 passages")

# Load and encode once at startup; each request only embeds its question.
model = SentenceTransformer(MODEL_NAME)
passage_vectors = model.encode(
    PASSAGES, normalize_embeddings=True, convert_to_numpy=True,
    show_progress_bar=False
).astype("float32")

app = FastAPI(title="RAG Colosseum", version="1.0.0")
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

def tokenize(s: str):
    return set(re.findall(r"[a-z0-9]+", s.lower()))

def extract_answer(question: str, passages: List[str]) -> str:
    """
    Extractive answer construction. The passages are treated strictly as
    untrusted reference data, never as instructions to the service.
    """
    q_terms = tokenize(question)
    scored = []
    for passage in passages:
        # Split on sentence boundaries; retain concise factual sentences.
        sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
        for sent in sentences:
            terms = tokenize(sent)
            overlap = len(q_terms & terms)
            # Favor direct lexical evidence, but don't require exact phrasing.
            score = overlap / max(1, len(q_terms))
            if overlap:
                scored.append((score, sent.strip()))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    chosen = []
    seen = set()
    for _, sent in scored:
        key = sent.lower()
        if key not in seen:
            chosen.append(sent)
            seen.add(key)
        if len(chosen) >= 3:
            break
    if not chosen:
        # If the query is semantically close but lexical overlap is low,
        # return the most relevant passage as a short source-grounded excerpt.
        return passages[0][:500]
    return " ".join(chosen)

@app.get("/")
def root():
    return {"service": "RAG Colosseum", "status": "ready", "endpoint": "POST /query"}

@app.get("/health")
def health():
    return {"status": "ok", "passages": len(PASSAGES), "embedding_model": MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    qv = model.encode([question], normalize_embeddings=True, convert_to_numpy=True,
                      show_progress_bar=False).astype("float32")[0]
    sims = passage_vectors @ qv
    order = np.argsort(-sims)
    best = float(sims[order[0]])

    # Conservative abstention for unrelated or unsupported questions.
    # Semantic similarity is only a retrieval signal, not proof of answerability.
    if best < 0.27:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.15)

    k = min(req.top_k, len(PASSAGES))
    selected = [int(i) for i in order[:k]]
    selected_scores = [round(max(0.0, min(1.0, float(sims[i]))), 4) for i in selected]

    # Reject weak matches even if the best passage barely crosses threshold.
    if best < 0.31:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.25)

    answer = extract_answer(question, [PASSAGES[i] for i in selected])
    # Confidence is bounded and based on the leading semantic match.
    confidence = max(0.61, min(0.96, 0.61 + (best - 0.31) * 0.8))
    return QueryResponse(
        answer=answer,
        sources=selected,
        scores=selected_scores,
        confidence=round(confidence, 3)
    )
