import os
import fitz
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------- SETTINGS ----------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 500
OVERLAP = 100
PDF_FOLDER = "/Users/dharshini/PycharmProjects/embeddingfor100books/.venv/clean_research_papers"            # folder containing PDFs
METADATA_JSON = "metadata.json"


# ---------------- HELPER FUNCTIONS ----------------
def extract_text_by_page(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks: list[str], model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)


# ---------------- PROCESS ALL PDFs ----------------
def process_pdfs(pdf_folder: str):
    model = SentenceTransformer(MODEL_NAME)
    all_data = []

    pdf_files = sorted(
        [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    )
    if not pdf_files:
        print(f"No PDFs found in folder '{pdf_folder}'")
        return None

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")
        pages = extract_text_by_page(pdf_path)

        for page_data in pages:
            page_num = page_data["page"]
            chunks = chunk_text(page_data["text"])
            if not chunks:
                continue

            embeddings = embed_chunks(chunks, model)

            for chunk, embedding in zip(chunks, embeddings):
                all_data.append({
                    "pdf_name": os.path.basename(pdf_path),
                    "page_number": page_num,
                    "chunk_text": chunk,
                    "embedding": embedding.tolist()  # convert numpy array to list for JSON
                })

    # Save all data (chunk + vector + metadata) to JSON
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"\nTotal chunks processed: {len(all_data)}")
    print(f"Metadata + embeddings saved to: {METADATA_JSON}")

    return all_data


# ---------------- MAIN ----------------
if __name__ == "__main__":
    all_data = process_pdfs(PDF_FOLDER)
    if all_data:
        print(f"Total chunks stored: {len(all_data)}")
