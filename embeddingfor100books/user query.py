import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_INDEX_PATH = "faiss.index"
METADATA_JSON = "metadata.json"  # JSON with chunk + embedding + metadata

# ---------------- LOAD FAISS & METADATA ----------------
def load_index(index_path: str = FAISS_INDEX_PATH) -> faiss.IndexFlatIP:
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors.")
    return index

def load_metadata(metadata_path: str = METADATA_JSON) -> list[dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata

# ---------------- EMBED QUERY ----------------
def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    embedding = model.encode([query], normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)

# ---------------- SEARCH ----------------
def search(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
    threshold: float = 0.5
) -> list[dict]:
    query_vector = embed_query(query, model)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < threshold:
            continue
        result = {
            "pdf_name": metadata[idx]["pdf_name"],
            "page_number": metadata[idx]["page_number"],
            "similarity_score": float(score),
            "chunk_text": metadata[idx]["chunk_text"],
        }
        results.append(result)
    return results

# ---------------- DISPLAY RESULTS ----------------
def display_results(results: list[dict]):
    if not results:
        print("\nNo results found above the similarity threshold.")
        return

    print(f"\n{'=' * 70}")
    print(f"Found {len(results)} result(s)")
    print(f"{'=' * 70}")

    for i, result in enumerate(results, 1):
        print(f"\n[Result {i}]")
        print(f"  PDF Name       : {result['pdf_name']}")
        print(f"  Page Number    : {result['page_number']}")
        print(f"  Similarity     : {result['similarity_score']:.4f}")
        print(f"  Relevant Text  :")
        print(f"  {'-' * 60}")
        print(f"  {result['chunk_text'][:500]}{'...' if len(result['chunk_text']) > 500 else ''}")
        print(f"  {'-' * 60}")

# ---------------- RUN PIPELINE ----------------
def run_search_pipeline(top_k: int = 5, threshold: float = 0.5):
    print("Loading model and index...")
    model = SentenceTransformer(MODEL_NAME)
    index = load_index()
    metadata = load_metadata()

    print("\nPDF Search Ready. Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting search.")
            break
        if not query:
            print("Please enter a valid query.")
            continue

        results = search(query, index, metadata, model, top_k=top_k, threshold=threshold)
        display_results(results)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    run_search_pipeline(top_k=5, threshold=0.5)
