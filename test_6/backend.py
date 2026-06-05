import io
import json
import os
import pickle
import re
import shutil
import fitz
import numpy as np
import faiss
from bson import ObjectId
from contextlib import asynccontextmanager
from datetime import datetime
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─── Configuration ─────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_STORE_FILE = "faiss_store.pkl"  # single combined file
UPLOAD_DIR = "uploaded_pdfs"
CHUNK_SIZE = 400
OVERLAP = 80
MIN_WORDS = 50
MIN_SCORE = 0.68

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── MongoDB ───────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["research_papers"]
collection = db["pdf_files"]

# ─── Global state (loaded once at startup) ─────────────────────
model = None
index = None
metadata = []


# ══════════════════════════════════════════════════════════════
#  STARTUP — load model + index into memory
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when server starts. Loads model and FAISS index."""
    global model, index, metadata

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model ready")

    if os.path.exists(FAISS_STORE_FILE):
        with open(FAISS_STORE_FILE, "rb") as f:
            store = pickle.load(f)
        index = faiss.deserialize_index(
            np.frombuffer(store["index_bytes"], dtype=np.uint8))
        metadata = store["metadata"]
        print(f"✓ FAISS store loaded — {index.ntotal} vectors, {len(metadata)} records")
    else:
        print("⚠  faiss_store.pkl not found.")
        print("   Run  python build_index.py  first.")

    print("\nServer ready at http://localhost:8000\n")
    yield


# ─── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="PDF Semantic Search",
    version="2.0.0",
    lifespan=lifespan
)

# CORS — allows the HTML frontend (served from port 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
#  TEXT HELPERS  (used for upload auto-indexing)
# ══════════════════════════════════════════════════════════════

CITATION_PATTERNS = [
    r"\b(19|20)\d{2}\b", r"arxiv\s*:\s*\d+\.\d+",
    r"http[s]?://", r"doi\.org",
    r"proceedings of", r"journal of",
    r"pp\.\s*\d+", r"vol\.\s*\d+",
    r"et al\.",
]


def is_reference_page(text: str) -> bool:
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
    dots = len(re.findall(r'\.{4,}', text))
    lines = [l for l in text.split('\n') if l.strip()]
    return bool(lines) and (dots / max(len(lines), 1)) > 0.40


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text: str) -> list:
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        chunk = " ".join(words[start:start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


def process_pdf_to_records(filepath: str, pdf_id: str, pdf_name: str) -> list:
    """
    Extract → filter → chunk → embed one PDF.
    Returns list of (embedding, metadata_dict) tuples.
    Used during upload to add a new PDF to the existing index.
    """
    if not os.path.exists(filepath):
        return []

    doc = fitz.open(filepath)
    records = []

    for page_num in range(len(doc)):
        raw = doc[page_num].get_text()
        if is_reference_page(raw) or is_toc_page(raw):
            continue
        cleaned = clean_text(raw)
        if len(cleaned.split()) < MIN_WORDS:
            continue

        chunks = chunk_text(cleaned)
        embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        for chunk, emb in zip(chunks, embeddings):
            records.append({
                "embedding": emb.astype(np.float32).tolist(),
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "page_number": page_num + 1,
                "chunk_text": chunk
            })

    doc.close()
    return records


def rebuild_faiss_index(all_records: list):
    """
    Rebuild the FAISS index from a list of records and persist both files.
    Called after a new PDF is uploaded.
    all_records: list of dicts with 'embedding' + metadata fields.
    """
    global index, metadata

    if not all_records:
        index = None
        metadata = []
        if os.path.exists(FAISS_STORE_FILE):
            os.remove(FAISS_STORE_FILE)
        return

    embs = np.array([r["embedding"] for r in all_records], dtype=np.float32)
    metadata = [{k: v for k, v in r.items() if k != "embedding"} for r in all_records]

    new_index = faiss.IndexFlatIP(embs.shape[1])
    new_index.add(embs)
    index = new_index

    store = {
        "index_bytes": faiss.serialize_index(index).tobytes(),
        "metadata": metadata
    }
    with open(FAISS_STORE_FILE, "wb") as f:
        pickle.dump(store, f)


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 1 — GET /api/stats
# ══════════════════════════════════════════════════════════════
@app.get("/api/stats")
def get_stats():
    """
    Returns system statistics.
    Frontend uses this to display PDF count and chunk count in the header.
    """
    return {
        "total_pdfs": collection.count_documents({}),
        "total_chunks": len(metadata),
        "index_vectors": index.ntotal if index else 0,
        "index_ready": index is not None,
        "model": MODEL_NAME
    }


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 2 — GET /api/pdfs
# ══════════════════════════════════════════════════════════════
@app.get("/api/pdfs")
def list_pdfs():
    """
    Returns all PDF records stored in MongoDB.
    Frontend uses this to populate the PDF Library tab and sidebar filter.
    """
    records = list(collection.find(
        {},
        {"_id": 1, "filename": 1, "filepath": 1, "uploaded_at": 1, "num_chunks": 1}
    ))
    return {
        "pdfs": [
            {
                "id": str(r["_id"]),
                "filename": r.get("filename", ""),
                "filepath": r.get("filepath", ""),
                "uploaded_at": str(r.get("uploaded_at", "")),
                "num_chunks": r.get("num_chunks", 0)
            }
            for r in records
        ],
        "total": len(records)
    }


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 3 — POST /api/search
# ══════════════════════════════════════════════════════════════
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    pdf_filter: str = ""  # empty = search all PDFs


@app.post("/api/search")
def search(req: SearchRequest):
    """
    Main semantic search endpoint.

    Steps:
      1. Embed the user query using the same model
      2. FAISS inner-product search → top candidates
      3. Check best score against MIN_SCORE threshold
         — if even the best result is below threshold, return empty
         — this means the query is completely unrelated to any paper
      4. Optionally filter by PDF name
      5. Deduplicate using first-80-char fingerprint
      6. Return top_k results
    """
    if index is None or not metadata:
        raise HTTPException(
            status_code=503,
            detail="Search index not ready. Run python build_index.py first."
        )
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Embed query — same model and normalisation as indexing
    query_vec = model.encode([req.query.strip()], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    # Search FAISS — fetch 3× candidates so filtering still gives top_k
    scores, indices = index.search(query_vec, req.top_k * 3)

    # ── Key fix: check the BEST score first ──────────────────────
    # If even the top result is below MIN_SCORE, the query is
    # completely unrelated to the indexed papers → return empty.
    # This prevents showing irrelevant results for random queries.
    best_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0
    if best_score < MIN_SCORE:
        return {
            "query": req.query,
            "results": [],
            "total": 0
        }

    results, seen = [], set()

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue

        # Discard individual results below threshold
        if float(score) < MIN_SCORE:
            continue

        m = metadata[idx]

        # Optional: filter to one specific PDF
        if req.pdf_filter and req.pdf_filter.lower() not in m["pdf_name"].lower():
            continue

        # Deduplicate by fingerprint (first 80 chars of chunk)
        fp = m["chunk_text"][:80]
        if fp in seen:
            continue
        seen.add(fp)

        results.append({
            "pdf_name": m["pdf_name"],
            "page_number": m["page_number"],
            "chunk_text": m["chunk_text"],
            "score": round(float(score), 4)
        })

        if len(results) >= req.top_k:
            break

    return {
        "query": req.query,
        "results": results,
        "total": len(results)
    }


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 4 — POST /api/upload
# ══════════════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a new PDF from the browser and automatically index it.

    Steps:
      1. Validate .pdf extension
      2. Check no duplicate filename in MongoDB
      3. Save file to uploaded_pdfs/ folder
      4. Insert record into MongoDB pdf_files collection
      5. Extract → chunk → embed the new PDF
      6. Load existing faiss_metadata.json
      7. Merge new records with existing ones
      8. Rebuild and save FAISS index
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed.")

    if collection.find_one({"filename": file.filename}):
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' already exists in the database."
        )

    # Save file to disk
    save_path = os.path.abspath(os.path.join(UPLOAD_DIR, file.filename))
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Insert into MongoDB
    doc = {
        "filename": file.filename,
        "filepath": save_path,
        "length": os.path.getsize(save_path),
        "uploaded_at": datetime.utcnow().isoformat() + "+00:00"
    }
    result = collection.insert_one(doc)
    pdf_id = str(result.inserted_id)

    # Extract, chunk, embed
    new_records = process_pdf_to_records(save_path, pdf_id, file.filename)

    if not new_records:
        collection.delete_one({"_id": result.inserted_id})
        os.remove(save_path)
        raise HTTPException(
            status_code=422,
            detail="No readable text found in this PDF."
        )

    # Update chunk count in MongoDB
    collection.update_one(
        {"_id": result.inserted_id},
        {"$set": {"num_chunks": len(new_records)}}
    )

    # Load existing metadata + merge + rebuild FAISS
    existing_records = []
    if os.path.exists(FAISS_STORE_FILE):
        with open(FAISS_METADATA_FILE, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        # Rebuild existing records with dummy embeddings for merging
        # We re-embed only the new PDF — existing vectors stay in the index
        # So we load the existing index and add new vectors on top
        global index, metadata

        existing_embs = faiss.read_index(FAISS_INDEX_FILE) if os.path.exists(FAISS_INDEX_FILE) else None
        new_embs = np.array([r["embedding"] for r in new_records], dtype=np.float32)
        new_meta = [{k: v for k, v in r.items() if k != "embedding"} for r in new_records]

        if existing_embs and existing_embs.ntotal > 0:
            # Add new vectors to existing index
            existing_embs.add(new_embs)
            index = existing_embs
            metadata = existing_meta + new_meta
        else:
            # No existing index — just create fresh
            dim = new_embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(new_embs)
            metadata = new_meta

        store = {
            "index_bytes": faiss.serialize_index(index).tobytes(),
            "metadata": metadata
        }
        with open(FAISS_STORE_FILE, "wb") as f:
            pickle.dump(store, f)

    else:
        # First upload ever — build index from scratch
        rebuild_faiss_index(new_records)

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "pdf_id": pdf_id,
        "chunks": len(new_records),
        "total_pdfs": collection.count_documents({})
    }


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 5 — GET /api/pdf/{pdf_id}
# ══════════════════════════════════════════════════════════════
@app.get("/api/pdf/{pdf_id}")
def serve_pdf(pdf_id: str):
    """
    Serves the actual PDF file so the browser can open it inline.
    Called when the user clicks a PDF name in search results or library.
    Content-Disposition: inline  →  opens in browser tab, not a download.
    """
    try:
        obj_id = ObjectId(pdf_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF id.")

    record = collection.find_one({"_id": obj_id})
    if not record:
        raise HTTPException(status_code=404, detail="PDF not found.")

    filepath = record.get("filepath", "")
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    filename = record.get("filename", "document.pdf")
    return FileResponse(
        path=filepath,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"{filename}\""}
    )