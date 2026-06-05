import io
import json
import os
import pickle
import re
import fitz  # PyMuPDF
import numpy as np
import faiss
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_STORE_FILE = "faiss_store.pkl"  # single combined file
CHUNK_SIZE = 400  # words per chunk
OVERLAP = 80  # overlapping words
MIN_WORDS = 50  # skip pages shorter than this

# ─── MongoDB ───────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["research_papers"]
collection = db["pdf_files"]

# ─── Citation / TOC detection patterns ────────────────────────
CITATION_PATTERNS = [
    r"\b(19|20)\d{2}\b",
    r"arxiv\s*:\s*\d+\.\d+",
    r"http[s]?://",
    r"doi\.org",
    r"proceedings of",
    r"journal of",
    r"pp\.\s*\d+",
    r"vol\.\s*\d+",
    r"et al\.",
]


# ══════════════════════════════════════════════════════════════
#  TEXT HELPERS
# ══════════════════════════════════════════════════════════════

def is_reference_page(text: str) -> bool:
    """Return True if the page looks like a bibliography / reference list."""
    lines = [l.strip() for l in text.split(".") if l.strip()]
    if re.search(r"^\s*references\s*$", text[:200], re.IGNORECASE | re.MULTILINE):
        return True
    if not lines:
        return False
    count = sum(
        1 for l in lines
        if any(re.search(p, l, re.IGNORECASE) for p in CITATION_PATTERNS)
    )
    return (count / len(lines)) >= 0.55


def is_toc_page(text: str) -> bool:
    """Return True if the page looks like a table of contents (lots of dots)."""
    dots = len(re.findall(r'\.{4,}', text))
    lines = [l for l in text.split('\n') if l.strip()]
    return bool(lines) and (dots / max(len(lines), 1)) > 0.40


def clean_text(text: str) -> str:
    """Remove newlines, unicode math symbols, and extra spaces."""
    text = text.replace("\n", " ")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # strip non-ASCII (math symbols)
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str) -> list:
    """Split cleaned text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


# ══════════════════════════════════════════════════════════════
#  CORE PIPELINE
# ══════════════════════════════════════════════════════════════

def process_and_build():
    """
    Full pipeline in one function:
      MongoDB → extract → filter → chunk → embed → FAISS → save
    """

    # ── Load embedding model ────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    print("Please wait...\n")
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model ready\n")

    # ── Fetch all PDF records from MongoDB ──────────────────────
    pdf_records = list(collection.find())

    if not pdf_records:
        print("No documents found in 'pdf_files' collection.")
        print("Please add PDFs to MongoDB first.")
        return

    print(f"Found {len(pdf_records)} PDFs in MongoDB.\n")
    print("=" * 60)

    all_embeddings = []  # numpy arrays — fed into FAISS
    all_metadata = []  # dicts — saved to faiss_metadata.json

    # ── Process each PDF ────────────────────────────────────────
    for record in pdf_records:
        pdf_id = str(record["_id"])
        pdf_name = record.get("filename", "unknown.pdf")
        filepath = record.get("filepath", "")

        print(f"Processing : {pdf_name}")

        if not filepath or not os.path.exists(filepath):
            print(f"  ✗ File not found at: {filepath} — skipping\n")
            continue

        # Open PDF
        doc = fitz.open(filepath)
        chunk_count = 0
        skip_ref = 0
        skip_toc = 0
        skip_short = 0

        for page_num in range(len(doc)):
            raw_text = doc[page_num].get_text()

            # ── Filter out non-content pages ────────────────────
            if is_reference_page(raw_text):
                skip_ref += 1
                continue
            if is_toc_page(raw_text):
                skip_toc += 1
                continue

            cleaned = clean_text(raw_text)

            if len(cleaned.split()) < MIN_WORDS:
                skip_short += 1
                continue

            # ── Chunk the page text ─────────────────────────────
            chunks = chunk_text(cleaned)
            if not chunks:
                continue

            # ── Embed all chunks from this page in one batch ────
            embeddings = model.encode(
                chunks,
                normalize_embeddings=True,  # L2 normalise → dot product = cosine
                show_progress_bar=False
            )

            # ── Store embedding + metadata together ─────────────
            for chunk, emb in zip(chunks, embeddings):
                all_embeddings.append(emb.astype(np.float32))
                all_metadata.append({
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "page_number": page_num + 1,
                    "chunk_text": chunk
                })
                chunk_count += 1

        total_pages = len(doc)  # save page count BEFORE closing
        doc.close()
        print(
            f"  Pages: {total_pages}  |  Skipped ref:{skip_ref} toc:{skip_toc} short:{skip_short}  |  Chunks: {chunk_count}\n")

    # ── Check we have something to index ────────────────────────
    if not all_embeddings:
        print("No chunks generated. Check your PDF filepaths.")
        return

    total_chunks = len(all_embeddings)
    print("=" * 60)
    print(f"Total chunks to index : {total_chunks}")

    # ── Build FAISS index ────────────────────────────────────────
    print("Building FAISS IndexFlatIP...")
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    dimension = embeddings_np.shape[1]  # 768 for bge-base

    index = faiss.IndexFlatIP(dimension)  # inner product = cosine similarity
    index.add(embeddings_np)  # add all vectors at once

    print(f"  Vectors in index    : {index.ntotal}")
    print(f"  Embedding dimension : {dimension}")

    # ── Combine index + metadata into ONE file ───────────────────
    # FAISS is serialised to bytes using faiss.serialize_index().
    # Those bytes + the metadata list are packed together in a
    # single pickle file.  Position mapping is preserved:
    #   index row 0  →  all_metadata[0]
    #   index row 1  →  all_metadata[1]  ... and so on.
    index_bytes = faiss.serialize_index(index).tobytes()
    store = {
        "index_bytes": index_bytes,  # raw FAISS binary
        "metadata": all_metadata  # list of dicts (no embeddings)
    }
    with open(FAISS_STORE_FILE, "wb") as f:
        pickle.dump(store, f)
    print(f"  ✓ Saved combined store : {FAISS_STORE_FILE}")
    print(f"    — vectors  : {index.ntotal}")
    print(f"    — metadata : {len(all_metadata)} records")

    print("\n" + "=" * 60)
    print("✓ Index build complete!")
    print(f"  Total PDFs processed : {len(pdf_records)}")
    print(f"  Total chunks indexed : {total_chunks}")
    print(f"  Combined store file  : {FAISS_STORE_FILE}")
    print("=" * 60)
    print("\nNext step: uvicorn backend:app --reload --port 8000")


# ─── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    process_and_build()