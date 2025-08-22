import os, json, argparse, pathlib
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
import faiss
from openai import OpenAI
from utils import l2_normalize

def load_index(index_dir: str):
    index_path = pathlib.Path(index_dir) / "faiss.index"
    meta_path = pathlib.Path(index_dir) / "meta.jsonl"
    dims_path = pathlib.Path(index_dir) / "dims.json"
    if not index_path.exists() or not meta_path.exists() or not dims_path.exists():
        raise RuntimeError("Faltan archivos del índice. Ejecuta primero index.py")

    with open(dims_path, "r", encoding="utf-8") as fr:
        meta = json.load(fr)
    dims = meta["dims"]
    model = meta["model"]

    index = faiss.read_index(str(index_path))

    metas = []
    with open(meta_path, "r", encoding="utf-8") as fr:
        for line in fr:
            metas.append(json.loads(line))

    return index, metas, dims, model

def embed_query(client: OpenAI, text: str, model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    vec = l2_normalize(vec)
    return vec.reshape(1, -1)

def top_k(index, qvec: np.ndarray, k: int = 3):
    scores, idxs = index.search(qvec, k)
    return scores[0], idxs[0]

def format_context(items: List[Dict], max_chars: int = 1200) -> str:
    acc = ""
    for it in items:
        chunk = f"[Fuente: {it['source']} | chunk {it['chunk_id']}]:\n{it['text']}\n\n"
        if len(acc) + len(chunk) > max_chars:
            break
        acc += chunk
    return acc.strip()

def generate_answer(client: OpenAI, question: str, context: str) -> str:
    prompt = f"""
Eres un asistente conciso. Responde a la pregunta basándote **exclusivamente** en el contexto.
Si la respuesta no está en el contexto, di que no hay información suficiente.

Pregunta: {question}

Contexto:
{context}
"""
    # Usa el modelo de tu preferencia (económico y capaz):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default=".indexes")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--answer", action="store_true", help="Generar respuesta con LLM usando el contexto recuperado.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno (.env).")

    client = OpenAI()
    index, metas, dims, embed_model = load_index(args.index_dir)
    
    SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.70"))


    # embedding de la consulta
    qvec = embed_query(client, args.question, embed_model)

    # búsqueda top-K
    scores, idxs = top_k(index, qvec, k=args.k)
    hits = []
    for score, idx in zip(scores, idxs):
        if idx == -1:
            continue
        m = metas[int(idx)]
        m["_score"] = float(score)
        hits.append(m)

    print("\n=== Top-K Chunks ===")
    for h in hits:
        print(f"* score={h['_score']:.3f} | {h['source']}#{h['chunk_id']}")
    print()
    
    # ✅ Guard-rail de tokens
    if not hits or hits[0]["_score"] < SIM_THRESHOLD:
        print("=== Respuesta (RAG) ===")
        print(f"* score={h['_score']:.3f} | {h['source']}#{h['chunk_id']}")
        print("No encontré información suficiente en los documentos.")
        return 

    if args.answer:
        ctx = format_context(hits, max_chars=1200)
        answer = generate_answer(client, args.question, ctx)
        print("=== Respuesta (RAG) ===")
        print(answer)

if __name__ == "__main__":
    main()
