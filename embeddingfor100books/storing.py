import faiss
import numpy as np
import json

# ---------------- SETTINGS ----------------
FAISS_INDEX_PATH = "faiss.index"
METADATA_JSON = "metadata.json"  # your JSON from chunk+embed

# ---------------- LOAD EMBEDDINGS ----------------
def load_embeddings(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = [np.array(item["embedding"], dtype=np.float32) for item in data]
    embeddings_matrix = np.array(embeddings, dtype=np.float32)
    return embeddings_matrix, data

# ---------------- BUILD & SAVE FAISS ----------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (works with normalized embeddings)
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.IndexFlatIP, path: str = FAISS_INDEX_PATH):
    faiss.write_index(index, path)
    print(f"FAISS index saved to: {path}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    embeddings_matrix, metadata = load_embeddings(METADATA_JSON)
    print(f"Loaded {embeddings_matrix.shape[0]} embeddings with dimension {embeddings_matrix.shape[1]}")

    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings_matrix)
    save_faiss_index(index)

    print(f"Total vectors stored in FAISS: {index.ntotal}")
