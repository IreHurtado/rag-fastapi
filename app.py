# app.py — API mínima para consultar índice RAG
import os, json, pathlib
from typing import List, Dict
import numpy as np
import faiss
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from fastapi import HTTPException, Request, Response
from datetime import date



MAX_REQ_PER_IP = 4
IP_COUNTS = defaultdict(int)
LAST_RESET = date.today()  


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Falta OPENAI_API_KEY en .env")

client = OpenAI()

SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.70"))


INDEX_DIR = ".indexes"
index_path = pathlib.Path(INDEX_DIR) / "faiss.index"
meta_path  = pathlib.Path(INDEX_DIR) / "meta.jsonl"
dims_path  = pathlib.Path(INDEX_DIR) / "dims.json"

if not (index_path.exists() and meta_path.exists() and dims_path.exists()):
    raise RuntimeError("No existe .indexes completo. Ejecuta primero: python src/index.py")

index = faiss.read_index(str(index_path))

metas = []
with open(meta_path, "r", encoding="utf-8") as fr:
    for line in fr:
        metas.append(json.loads(line))

dims_meta = json.load(open(dims_path, "r", encoding="utf-8"))
EMBED_MODEL = dims_meta["model"]       

def reset_if_new_day():
    global LAST_RESET, IP_COUNTS
    today = date.today()
    if today != LAST_RESET:
        print("[RATE] Nuevo día, reseteando contadores...")
        IP_COUNTS = defaultdict(int)
        LAST_RESET = today

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = l2_normalize(v).reshape(1, -1)
    return v  # (1, D)

def search_topk(qvec: np.ndarray, k: int = 3):
    scores, idxs = index.search(qvec, k)
    return scores[0], idxs[0]


def build_context(items: List[Dict], max_chars: int = 1200) -> str:
    acc = ""
    for it in items:
        chunk = f"[Fuente: {it['source']} | chunk {it['chunk_id']}]\n{it['text']}\n\n"
        if len(acc) + len(chunk) > max_chars:
            break
        acc += chunk
    return acc.strip()


def rag_answer(question: str, context: str) -> str:
    prompt = f"""
Eres un asistente conciso. Responde basándote SOLO en el contexto.
Si no hay información suficiente en el contexto, dilo claramente.

Pregunta: {question}

Contexto:
{context}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------- FastAPI ----------
app = FastAPI(title="RAG Demo API")

# CORS abierto para dev local 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción especificar dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskIn(BaseModel):
    question: str
    k: int = 3
    answer: bool = True  

@app.post("/ask")
def ask(body: AskIn, request: Request, response: Response):
    ip = request.client.host or "unknown"
    reset_if_new_day()

    # 1) Chequeo de límite
    if IP_COUNTS[ip] >= MAX_REQ_PER_IP:
        response.headers["X-RateLimit-Limit"] = str(MAX_REQ_PER_IP)
        response.headers["X-RateLimit-Remaining"] = "0"
        raise HTTPException(status_code=429, detail="Has alcanzado el límite de 4 preguntas por IP.")

    # 2) Contabilizar esta petición
    IP_COUNTS[ip] += 1
    remaining = MAX_REQ_PER_IP - IP_COUNTS[ip]
    response.headers["X-RateLimit-Limit"] = str(MAX_REQ_PER_IP)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))

    # 3) Búsqueda semántica
    qvec = embed_query(body.question)
    scores, idxs = search_topk(qvec, k=body.k)

    hits = []
    for s, i in zip(scores, idxs):
        if i == -1:
            continue
        m = metas[int(i)].copy()
        m["_score"] = float(s)
        hits.append(m)
        
    # 4) Guard-rail de tokens    
    if not hits or hits[0]["_score"] < SIM_THRESHOLD:
        return {
            "answer": "No encontré información suficiente en los documentos.",
            "hits": []
        }

    payload = {"question": body.question, "hits": hits}
    if body.answer:
        ctx = build_context(hits)
        payload["answer"] = rag_answer(body.question, ctx)

    return payload

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")  
