import os
import fitz
import json

PDF_FOLDER = "Books"
CHUNK_SIZE = 500
OVERLAP = 100
OUTPUT_JSON = "chunks.json"


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


data = []

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, file)
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):

            page = doc.load_page(page_num)
            text = page.get_text()

            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

            for chunk in chunks:

                data.append({
                    "book": file,
                    "page": page_num + 1,
                    "text": chunk
                })


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Chunking completed")