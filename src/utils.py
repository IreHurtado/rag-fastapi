import os, math, re, json
from typing import List, Dict, Iterable, Tuple

def clean_text(s: str) -> str:
    s = s.replace("\u200b", " ").strip()
    return re.sub(r"\s+", " ", s)

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    text = clean_text(text)
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += max(1, (chunk_size - chunk_overlap))
    return chunks

def l2_normalize(vec):
    import numpy as np
    v = np.array(vec, dtype="float32")
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
