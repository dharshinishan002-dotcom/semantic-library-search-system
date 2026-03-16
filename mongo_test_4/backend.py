import json
import os
import re
import shutil
import fitz
import numpy as np
import faiss
from bson import ObjectId
from datetime import datetime
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_INDEX_FILE = "faiss_index.index"
FAISS_METADATA_FILE = "faiss_metadata.json"
METADATA_JSON = "metadata.json"
UPLOAD_DIR = "uploaded_pdfs"
CHUNK_SIZE = 400
OVERLAP = 80
MIN_WORDS = 50
MIN_SCORE = 0.45

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── MongoDB ───────────────────────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["research_papers"]
collection = db["pdf_files"]

# ─── Global state ──────────────────────────────────────────────────────────────
model = None
index = None
metadata = []


# ─── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup: loads model, FAISS index, and metadata into memory."""
    global model, index, metadata

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model ready")

    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_METADATA_FILE, "r") as f:
            metadata = json.load(f)
        print(f"✓ FAISS index ready — {index.ntotal} vectors")
    else:
        print("⚠  FAISS index not found. Run pdf_processing.py + faiss_store.py first.")

    print("\nServer ready at http://localhost:8000\n")
    yield  # everything after yield runs on shutdown (nothing needed here)


# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PDF Semantic Search", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Text helpers ──────────────────────────────────────────────────────────────
CITATION_PATTERNS = [
    r"\b(19|20)\d{2}\b", r"arxiv\s*:\s*\d+\.\d+",
    r"http[s]?://", r"doi\.org",
    r"proceedings of", r"journal of",
    r"pp\.\s*\d+", r"vol\.\s*\d+",
    r"et al\.",
]


def is_reference_page(text):
    lines = [l.strip() for l in text.split(".") if l.strip()]
    if re.search(r"^\s*references\s*$", text[:200], re.IGNORECASE | re.MULTILINE):
        return True
    if not lines:
        return False
    count = sum(1 for l in lines
                if any(re.search(p, l, re.IGNORECASE) for p in CITATION_PATTERNS))
    return (count / len(lines)) >= 0.55


def is_toc_page(text):
    dots = len(re.findall(r'\.{4,}', text))
    lines = [l for l in text.split('\n') if l.strip()]
    return bool(lines) and (dots / max(len(lines), 1)) > 0.40


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def chunk_text(text):
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        chunk = " ".join(words[start:start + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


def process_pdf(filepath, pdf_id, pdf_name):
    """Extract → filter → chunk → embed. Returns list of records."""
    if not os.path.exists(filepath):
        return []
    doc, records = fitz.open(filepath), []
    for page_num in range(len(doc)):
        raw = doc[page_num].get_text()
        if is_reference_page(raw) or is_toc_page(raw):
            continue
        cleaned = clean_text(raw)
        if len(cleaned.split()) < MIN_WORDS:
            continue
        chunks = chunk_text(cleaned)
        embeddings = model.encode(chunks, normalize_embeddings=True,
                                  show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            records.append({
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "page_number": page_num + 1,
                "chunk_text": chunk,
                "embedding": emb.tolist()
            })
    doc.close()
    return records


def rebuild_faiss(all_records):
    """Rebuild FAISS index from a list of records and persist to disk."""
    global index, metadata

    if not all_records:
        # No records left — create an empty placeholder
        index = None
        metadata = []
        # Remove index files so server knows index is empty
        for f in [FAISS_INDEX_FILE, FAISS_METADATA_FILE]:
            if os.path.exists(f):
                os.remove(f)
        with open(METADATA_JSON, "w") as f:
            json.dump([], f)
        return

    embs = np.array([r["embedding"] for r in all_records], dtype=np.float32)
    metadata = [{k: v for k, v in r.items() if k != "embedding"} for r in all_records]
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(METADATA_JSON, "w") as f:
        json.dump(all_records, f, indent=2)


# ══════════════════════════════════════════════════════
#  ENDPOINT 1 — GET /api/stats
# ══════════════════════════════════════════════════════
@app.get("/api/stats")
def get_stats():
    return {
        "total_pdfs": collection.count_documents({}),
        "total_chunks": len(metadata),
        "index_vectors": index.ntotal if index else 0,
        "index_ready": index is not None,
        "model": MODEL_NAME
    }


# ══════════════════════════════════════════════════════
#  ENDPOINT 2 — GET /api/pdfs
# ══════════════════════════════════════════════════════
@app.get("/api/pdfs")
def list_pdfs():
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


# ══════════════════════════════════════════════════════
#  ENDPOINT 3 — POST /api/search
# ══════════════════════════════════════════════════════
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    pdf_filter: str = ""


@app.post("/api/search")
def search(req: SearchRequest):
    if index is None or not metadata:
        raise HTTPException(status_code=503,
                            detail="Index not ready. Run pdf_processing.py + faiss_store.py first.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    qvec = model.encode([req.query.strip()], normalize_embeddings=True)
    qvec = np.array(qvec, dtype=np.float32)
    scores, indices = index.search(qvec, req.top_k * 3)

    results, seen = [], set()

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        if float(score) < MIN_SCORE:
            continue
        m = metadata[idx]
        if req.pdf_filter and req.pdf_filter.lower() not in m["pdf_name"].lower():
            continue
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

    return {"query": req.query, "results": results, "total": len(results)}


# ══════════════════════════════════════════════════════
#  ENDPOINT 4 — POST /api/upload
# ══════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF → save to disk → insert MongoDB record →
    extract/chunk/embed → rebuild FAISS index.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files allowed.")

    if collection.find_one({"filename": file.filename}):
        raise HTTPException(status_code=409,
                            detail=f"'{file.filename}' already exists.")

    save_path = os.path.abspath(os.path.join(UPLOAD_DIR, file.filename))
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = {
        "filename": file.filename,
        "filepath": save_path,
        "length": os.path.getsize(save_path),
        "uploaded_at": datetime.utcnow().isoformat() + "+00:00"
    }
    result = collection.insert_one(doc)
    pdf_id = str(result.inserted_id)

    new_records = process_pdf(save_path, pdf_id, file.filename)

    if not new_records:
        collection.delete_one({"_id": result.inserted_id})
        os.remove(save_path)
        raise HTTPException(status_code=422,
                            detail="No readable text found in this PDF.")

    collection.update_one({"_id": result.inserted_id},
                          {"$set": {"num_chunks": len(new_records)}})

    existing = []
    if os.path.exists(METADATA_JSON):
        with open(METADATA_JSON, "r") as f:
            existing = json.load(f)

    rebuild_faiss(existing + new_records)

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "pdf_id": pdf_id,
        "chunks": len(new_records),
        "total_pdfs": collection.count_documents({})
    }


# ══════════════════════════════════════════════════════
#  ENDPOINT 5 — DELETE /api/delete/{pdf_id}
# ══════════════════════════════════════════════════════
@app.delete("/api/delete/{pdf_id}")
def delete_pdf(pdf_id: str):
    """
    Delete a PDF completely from:
      1. MongoDB  — removes the pdf_files document
      2. Disk     — deletes the physical PDF file (only if in uploaded_pdfs/)
      3. FAISS    — removes all chunks for this PDF and rebuilds the index
      4. metadata.json / faiss_metadata.json — updated automatically

    pdf_id: the MongoDB ObjectId string (e.g. "69b3bd6e0cee2c62c154e8e3")

    Response:
      {
        "message": "'2301.08028v4.pdf' deleted successfully.",
        "chunks_removed": 413,
        "total_pdfs": 99
      }
    """
    # ── Step 1: Find the record in MongoDB ─────────────────────────────────────
    try:
        obj_id = ObjectId(pdf_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF id format.")

    record = collection.find_one({"_id": obj_id})
    if not record:
        raise HTTPException(status_code=404,
                            detail=f"PDF with id '{pdf_id}' not found in MongoDB.")

    pdf_name = record.get("filename", "")
    filepath = record.get("filepath", "")

    # ── Step 2: Delete physical file from disk (only uploaded PDFs) ────────────
    # We only delete files inside the uploaded_pdfs/ folder.
    # Original PDFs on Desktop are left untouched.
    if filepath and UPLOAD_DIR in filepath and os.path.exists(filepath):
        os.remove(filepath)
        print(f"✓ Deleted file: {filepath}")
    else:
        print(f"ℹ  File not in upload dir — skipping disk delete: {filepath}")

    # ── Step 3: Remove from MongoDB ────────────────────────────────────────────
    collection.delete_one({"_id": obj_id})
    print(f"✓ Removed from MongoDB: {pdf_name}")

    # ── Step 4: Remove all chunks for this PDF from metadata.json ──────────────
    existing_records = []
    if os.path.exists(METADATA_JSON):
        with open(METADATA_JSON, "r") as f:
            existing_records = json.load(f)

    # Keep only chunks that belong to OTHER PDFs
    remaining_records = [r for r in existing_records if r["pdf_id"] != pdf_id]
    chunks_removed = len(existing_records) - len(remaining_records)

    print(f"✓ Removed {chunks_removed} chunks for '{pdf_name}' from index")

    # ── Step 5: Rebuild FAISS index without this PDF's chunks ──────────────────
    rebuild_faiss(remaining_records)
    print(f"✓ FAISS index rebuilt — {len(remaining_records)} vectors remaining")

    return {
        "message": f"'{pdf_name}' deleted successfully.",
        "chunks_removed": chunks_removed,
        "total_pdfs": collection.count_documents({})
    }


# ══════════════════════════════════════════════════════
#  ENDPOINT 6 — PUT /api/update/{pdf_id}
# ══════════════════════════════════════════════════════
@app.put("/api/update/{pdf_id}")
async def update_pdf(pdf_id: str, file: UploadFile = File(...)):
    """
    Replace an existing PDF with a new version.

    What it does:
      1. Find the existing PDF record in MongoDB by pdf_id
      2. Delete the old file from disk (if it was in uploaded_pdfs/)
      3. Save the new file to disk
      4. Update the MongoDB record (filename, filepath, length, updated_at)
      5. Remove all old chunks for this PDF from metadata
      6. Extract, chunk, embed the new PDF
      7. Rebuild FAISS index with updated chunks

    Use case: you have an updated version of a paper and want to
              replace the old one without deleting and re-uploading.

    Response:
      {
        "message": "'paper_v2.pdf' updated successfully.",
        "old_filename": "paper_v1.pdf",
        "new_filename": "paper_v2.pdf",
        "old_chunks": 87,
        "new_chunks": 95
      }
    """
    # ── Step 1: Find existing record ────────────────────────────────────────────
    try:
        obj_id = ObjectId(pdf_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF id format.")

    record = collection.find_one({"_id": obj_id})
    if not record:
        raise HTTPException(status_code=404,
                            detail=f"PDF with id '{pdf_id}' not found.")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files allowed.")

    old_filename = record.get("filename", "")
    old_filepath = record.get("filepath", "")

    # ── Step 2: Delete old file from disk (only if in uploaded_pdfs/) ──────────
    if old_filepath and UPLOAD_DIR in old_filepath and os.path.exists(old_filepath):
        os.remove(old_filepath)
        print(f"✓ Removed old file: {old_filepath}")

    # ── Step 3: Save new file to disk ────────────────────────────────────────────
    save_path = os.path.abspath(os.path.join(UPLOAD_DIR, file.filename))
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ── Step 4: Update MongoDB record ────────────────────────────────────────────
    collection.update_one(
        {"_id": obj_id},
        {"$set": {
            "filename": file.filename,
            "filepath": save_path,
            "length": os.path.getsize(save_path),
            "updated_at": datetime.utcnow().isoformat() + "+00:00"
        }}
    )

    # ── Step 5: Remove old chunks for this pdf_id ────────────────────────────────
    existing_records = []
    if os.path.exists(METADATA_JSON):
        with open(METADATA_JSON, "r") as f:
            existing_records = json.load(f)

    other_records = [r for r in existing_records if r["pdf_id"] != pdf_id]
    old_chunks = len(existing_records) - len(other_records)

    # ── Step 6: Process new PDF ──────────────────────────────────────────────────
    new_records = process_pdf(save_path, pdf_id, file.filename)

    if not new_records:
        # Rollback MongoDB update if extraction fails
        collection.update_one(
            {"_id": obj_id},
            {"$set": {"filename": old_filename, "filepath": old_filepath}}
        )
        os.remove(save_path)
        raise HTTPException(status_code=422,
                            detail="No readable text found in the new PDF.")

    # ── Step 7: Rebuild FAISS with updated chunks ────────────────────────────────
    collection.update_one({"_id": obj_id}, {"$set": {"num_chunks": len(new_records)}})
    rebuild_faiss(other_records + new_records)

    print(f"✓ Updated '{old_filename}' → '{file.filename}' ({old_chunks} → {len(new_records)} chunks)")

    return {
        "message": f"'{file.filename}' updated successfully.",
        "old_filename": old_filename,
        "new_filename": file.filename,
        "old_chunks": old_chunks,
        "new_chunks": len(new_records),
        "total_pdfs": collection.count_documents({})
    }


# ══════════════════════════════════════════════════════
#  ENDPOINT — GET /api/pdf/{pdf_id}
#  Serves the actual PDF file so browser can open it
# ══════════════════════════════════════════════════════
@app.get("/api/pdf/{pdf_id}")
def serve_pdf(pdf_id: str):
    """
    Returns the PDF file so the browser can open/view it.
    Called when user clicks a PDF name in search results or library.
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
        raise HTTPException(status_code=404,
                            detail="PDF file not found on disk.")

    # content_disposition_type="inline" tells the browser to DISPLAY
    # the PDF inside the tab instead of downloading it
    filename = record.get("filename", "document.pdf")
    return FileResponse(
        path=filepath,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename=\"{filename}\""
        }
    )