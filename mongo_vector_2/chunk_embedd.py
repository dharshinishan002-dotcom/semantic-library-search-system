import io
import json
import fitz
import numpy as np
from pymongo import MongoClient
import gridfs
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 500
OVERLAP = 100
METADATA_JSON = "metadata.json"

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["research_papers"]
fs = gridfs.GridFS(db)


def extract_text_by_page(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        pages.append({
            "page": page_num + 1,
            "text": text
        })

    doc.close()
    return pages


def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += CHUNK_SIZE - OVERLAP

    return chunks


def embed_chunks(chunks, model):
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def process_pdfs():

    model = SentenceTransformer(MODEL_NAME)
    all_data = []

    files = fs.find()

    for file in files:

        pdf_id = str(file._id)      # MongoDB ObjectId
        pdf_name = file.filename    # PDF filename

        print("Processing:", pdf_name)

        pdf_bytes = file.read()

        pages = extract_text_by_page(pdf_bytes)

        for page in pages:

            chunks = chunk_text(page["text"])

            if not chunks:
                continue

            embeddings = embed_chunks(chunks, model)

            for chunk, embedding in zip(chunks, embeddings):

                all_data.append({
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "page_number": page["page"],
                    "chunk_text": chunk,
                    "embedding": embedding.tolist()
                })

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)

    print("Total chunks:", len(all_data))


if __name__ == "__main__":
    process_pdfs()