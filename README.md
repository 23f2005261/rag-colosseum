# RAG Colosseum

FastAPI service for the 50-passage RAG Colosseum assignment.

## Deploy on Render
1. Push all four files (`app.py`, `requirements.txt`, `render.yaml`, `knowledge_base.json`) to a new GitHub repository.
2. In Render, create a new **Web Service** and connect that repository.
3. Use Python runtime, build command `pip install -r requirements.txt`, and start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.
4. Wait for the service to finish starting. The model downloads and passage embeddings are created during startup.
5. Submit the **base URL only** (for example `https://your-service.onrender.com`) in the assignment's Public Base URL field. The checker will call `/query`.

## API
`POST /query`
```json
{"question":"What does Raft do during a network partition?","top_k":3}
```
Response:
```json
{"answer":"...","sources":[0,2],"scores":[0.8,0.7],"confidence":0.8}
```

`GET /health` reports readiness.

## Notes
- The knowledge base is loaded from the supplied `knowledge_base.json`.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` and cosine similarity.
- The answer generator is extractive and uses retrieved passages as data only; it does not follow instructions found inside passages.
- This is a baseline implementation. Run the assignment's sample probes and tune the abstention thresholds / extraction logic against observed results before using a limited graded submission.
- Render free instances may cold-start and may have memory constraints because the embedding model downloads and loads at startup. Confirm `/health` is ready before submitting.
