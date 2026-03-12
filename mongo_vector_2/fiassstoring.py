import json
import numpy as np
import faiss

METADATA_JSON = "metadata.json"
FAISS_INDEX_FILE = "faiss_index.index"
FAISS_METADATA_FILE = "faiss_metadata.json"

def store_in_faiss():

    # Load metadata
    with open(METADATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    metadata = []

    for item in data:
        embeddings.append(item["embedding"])
        metadata.append({
            "pdf_id": item["pdf_id"],
            "pdf_name": item["pdf_name"],
            "page_number": item["page_number"],
            "chunk_text": item["chunk_text"]
        })

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings)

    # Save index
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save metadata
    with open(FAISS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Total vectors stored:", index.ntotal)


if __name__ == "__main__":
    store_in_faiss()