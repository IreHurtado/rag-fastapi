import os, json, argparse, pathlib
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI

from utils import chunk_text, l2_normalize

def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    step = 100  # batch simple
    embs = []
    for i in tqdm(range(0, len(texts), step), desc="Embeddings"):
        batch = texts[i:i+step]
        resp = client.embeddings.create(model=model, input=batch)
        for r in resp.data:
            embs.append(r.embedding)
    arr = np.array(embs, dtype="float32")
    # normalizar para usar IP como coseno
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=".indexes")
    parser.add_argument("--embed_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno (.env).")

    client = OpenAI()

    data_path = pathlib.Path(args.data_dir)
    out_path = pathlib.Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) Leer textos
    texts = []
    metas = []
    for p in sorted(data_path.glob("*.txt")):
        raw = p.read_text(encoding="utf-8")
        chunks = chunk_text(raw, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        for idx, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"source": str(p.name), "chunk_id": idx, "text": ch})

    if not texts:
        raise RuntimeError(f"No se encontraron .txt en {data_path}")

    # 2) Embeddings
    emb = embed_texts(client, texts, args.embed_model)
    dims = emb.shape[1]

    # 3) FAISS index (IP con vectores normalizados == coseno)
    index = faiss.IndexFlatIP(dims)
    index.add(emb)

    # 4) Guardar índice y metadatos
    faiss.write_index(index, str(out_path / "faiss.index"))
    with open(out_path / "meta.jsonl", "w", encoding="utf-8") as fw:
        for m in metas:
            fw.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(out_path / "dims.json", "w", encoding="utf-8") as fw:
        json.dump({"dims": int(dims), "model": args.embed_model}, fw, ensure_ascii=False)

    print(f"Index guardado en: {out_path}")

if __name__ == "__main__":
    main()
