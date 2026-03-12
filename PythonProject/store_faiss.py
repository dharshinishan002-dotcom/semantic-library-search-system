import json
import faiss
import numpy as np

INPUT_JSON = "metadata.json"
FAISS_INDEX = "books.index"

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

embeddings = [item["embedding"] for item in data]

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, FAISS_INDEX)

print("Stored in FAISS")