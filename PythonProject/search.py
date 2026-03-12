import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_INDEX = "books.index"
METADATA = "metadata.json"

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

index = faiss.read_index(FAISS_INDEX)

with open(METADATA, "r") as f:
    data = json.load(f)

query = input("Enter your question: ")

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")

D, I = index.search(query_embedding, 3)

print("\nTop Results:\n")

for i in range(len(I[0])):
    idx = I[0][i]
    score = D[0][i]


    result = data[idx]

    print("Book:", result["book"])
    print("Page:", result["page"])
    print("Similarity Score:", score)
    print("Text:", result["text"])
    print()