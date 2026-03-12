import json
from sentence_transformers import SentenceTransformer

INPUT_JSON = "chunks.json"
OUTPUT_JSON = "metadata.json"

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    embedding = model.encode(item["text"]).tolist()
    item["embedding"] = embedding


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f)

print("Embeddings created")