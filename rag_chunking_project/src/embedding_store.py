import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from chunking import recursive_chunking

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_database():

    chunks = recursive_chunking()

    print("Total chunks received:", len(chunks))

    # create embeddings
    embeddings = model.encode(
        chunks,
        batch_size=64,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    # create FAISS index
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("Vectors stored:", index.ntotal)

    # save index
    faiss.write_index(index, "faiss_index.bin")

    # save metadata
    metadata = []

    for i, chunk in enumerate(chunks):
        metadata.append({
            "chunk_id": i,
            "text": chunk
        })

    with open("chunk_metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Database saved successfully")


if __name__ == "__main__":
    build_vector_database()