import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_INDEX_FILE = "faiss_index.index"
METADATA_FILE = "faiss_metadata.json"

# Load model
model = SentenceTransformer(MODEL_NAME)

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_FILE)

# Load metadata
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def search_query(user_query, top_k=5):

    query_embedding = model.encode([user_query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    seen_chunks = set()

    for idx in indices[0]:

        if idx == -1 or idx >= len(metadata):
            continue

        data = metadata[idx]

        chunk = data["chunk_text"]

        # Remove duplicate chunks
        if chunk in seen_chunks:
            continue

        seen_chunks.add(chunk)

        results.append({
            "pdf_id": data["pdf_id"],
            "pdf_name": data["pdf_name"],
            "page_number": data["page_number"],
            "chunk_text": chunk
        })

    return results


if __name__ == "__main__":

    print("\nPDF Semantic Search Ready")

    while True:

        query = input("\nEnter your question (type 'exit' to stop): ").strip()

        if query.lower() == "exit":
            print("Exiting program...")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        results = search_query(query)

        print("\nTop Results:\n")

        if not results:
            print("No results found.")
            continue

        for i, r in enumerate(results, 1):

            print(f"Result {i}")
            print("PDF Name :", r["pdf_name"])
            print("Page     :", r["page_number"])
            print("Text     :", r["chunk_text"][:300])
            print("-" * 60)