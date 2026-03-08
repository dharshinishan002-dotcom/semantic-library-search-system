import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# load vector database
index = faiss.read_index("faiss_index.bin")

# load metadata
with open("chunk_metadata.json") as f:
    metadata = json.load(f)


def search(query, k=5):

    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, k)

    print("\nTop results:\n")

    for i in indices[0]:
        print(metadata[i]["text"])
        print("\n-----------------\n")


if __name__ == "__main__":

    query = input("Enter your question: ")

    search(query)