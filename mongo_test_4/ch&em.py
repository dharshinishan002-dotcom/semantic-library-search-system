import json
import os
import re
import fitz  # PyMuPDF
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 400  # Increased: more context per chunk
OVERLAP = 80  # Increased overlap
MIN_WORDS = 50  # Skip pages with fewer words than this
METADATA_JSON = "metadata.json"

# ─── MongoDB ───────────────────────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["research_papers"]
collection = db["pdf_files"]

# ─── Reference / Bibliography Detection ────────────────────────────────────────
# These patterns identify pages that are purely citations/references.
# Such pages cause every query to return bibliography entries as top results.
REFERENCE_SECTION_HEADERS = [
    r"^\s*references\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*works cited\s*$",
    r"^\s*citations\s*$",
]

# If a page has >= this fraction of lines that look like citations, skip it
CITATION_LINE_RATIO = 0.55


def is_reference_page(text: str) -> bool:
    """
    Return True if this page appears to be a bibliography/reference page.

    Detection strategy:
      1. Page starts with a known reference-section header, OR
      2. More than 55% of lines match citation patterns (author, year, URL, arXiv)
    """
    lines = [l.strip() for l in text.split(".") if l.strip()]

    # Check for section header at start of page
    first_200 = text[:200].lower()
    for pattern in REFERENCE_SECTION_HEADERS:
        if re.search(pattern, first_200, re.MULTILINE | re.IGNORECASE):
            return True

    if not lines:
        return False

    # Count lines that look like citation entries
    citation_patterns = [
        r"\b(19|20)\d{2}\b",  # Year like 1999, 2023
        r"arxiv\s*:\s*\d+\.\d+",  # arXiv ID
        r"http[s]?://",  # URL
        r"doi\.org",  # DOI
        r"proceedings of",  # Conference proceedings
        r"journal of",  # Journal name
        r"pp\.\s*\d+",  # Page numbers pp. 123
        r"vol\.\s*\d+",  # Volume number
        r"et al\.",  # et al.
        r"\.pdf$",  # PDF link
    ]

    citation_line_count = 0
    for line in lines:
        for pat in citation_patterns:
            if re.search(pat, line, re.IGNORECASE):
                citation_line_count += 1
                break

    ratio = citation_line_count / len(lines)
    return ratio >= CITATION_LINE_RATIO


def is_toc_page(text: str) -> bool:
    """
    Return True if the page looks like a Table of Contents.
    TOC pages are full of dots '......' and page numbers — not useful content.
    """
    dot_sequences = len(re.findall(r'\.{4,}', text))  # sequences of 4+ dots
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return False
    # If more than 40% of lines contain dot sequences → TOC page
    return (dot_sequences / max(len(lines), 1)) > 0.40


def clean_text(text: str) -> str:
    """
    Clean raw PDF text:
      - Replace newlines with spaces
      - Remove non-ASCII math/formula characters (e.g. 𝑀𝐴𝐷, 𝑋̃)
      - Collapse multiple spaces
    """
    text = text.replace("\n", " ")

    # Remove unicode math symbols (common in research PDFs, not useful for search)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_text_by_page(filepath: str) -> list:
    """
    Open a PDF and return cleaned text for each CONTENT page.
    Skips: reference pages, TOC pages, and pages with too little text.

    Returns:
        List of dicts: [{"page": int, "text": str}, ...]
    """
    if not os.path.exists(filepath):
        print(f"    ✗ File not found: {filepath}")
        return []

    doc = fitz.open(filepath)
    pages = []
    skipped = {"ref": 0, "toc": 0, "short": 0}

    for page_num in range(len(doc)):
        raw_text = doc[page_num].get_text()

        # ── Skip reference/bibliography pages ──────────────────────────────
        if is_reference_page(raw_text):
            skipped["ref"] += 1
            continue

        # ── Skip table-of-contents pages ───────────────────────────────────
        if is_toc_page(raw_text):
            skipped["toc"] += 1
            continue

        cleaned = clean_text(raw_text)

        # ── Skip pages with too little real content ─────────────────────────
        if len(cleaned.split()) < MIN_WORDS:
            skipped["short"] += 1
            continue

        pages.append({
            "page": page_num + 1,
            "text": cleaned
        })

    doc.close()

    if any(v > 0 for v in skipped.values()):
        print(f"    Skipped → ref:{skipped['ref']}  toc:{skipped['toc']}  short:{skipped['short']}")

    return pages


def chunk_text(text: str) -> list:
    """
    Split page text into overlapping word-based chunks.
    Uses CHUNK_SIZE=400 and OVERLAP=80 for better semantic context.
    """
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


def generate_embeddings(chunks: list, model: SentenceTransformer) -> np.ndarray:
    """
    Encode chunks to L2-normalised float32 embeddings.
    Normalisation makes dot-product equal to cosine similarity (for IndexFlatIP).
    """
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return np.array(embeddings, dtype=np.float32)


def process_pdfs():
    """
    Main pipeline:
      1. Read all records from MongoDB 'pdf_files' collection.
      2. Load each PDF from its filepath on disk.
      3. Filter out reference/TOC/short pages.
      4. Chunk and embed remaining content pages.
      5. Save to metadata.json.
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    print("Please wait...\n")
    model = SentenceTransformer(MODEL_NAME)
    all_data = []

    pdf_records = list(collection.find())

    if not pdf_records:
        print("No documents found in 'pdf_files' collection.")
        return

    print(f"Found {len(pdf_records)} PDFs in MongoDB.\n")
    print("=" * 60)

    for record in pdf_records:
        pdf_id = str(record["_id"])
        pdf_name = record["filename"]
        filepath = record["filepath"]

        print(f"Processing : {pdf_name}")

        pages = extract_text_by_page(filepath)

        if not pages:
            print(f"  → No usable content pages. Skipping.\n")
            continue

        chunk_count = 0

        for page in pages:
            chunks = chunk_text(page["text"])

            if not chunks:
                continue

            embeddings = generate_embeddings(chunks, model)

            for chunk, embedding in zip(chunks, embeddings):
                all_data.append({
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "page_number": page["page"],
                    "chunk_text": chunk,
                    "embedding": embedding.tolist()
                })
                chunk_count += 1

        print(f"  → Content pages: {len(pages)}  |  Chunks: {chunk_count}\n")

    # ── Save metadata.json ──────────────────────────────────────────────────────
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)

    print("=" * 60)
    print(f"✓ Done!")
    print(f"  Total PDFs    : {len(pdf_records)}")
    print(f"  Total chunks  : {len(all_data)}")
    print(f"  Saved to      : {METADATA_JSON}")
    print("=" * 60)
    print("\nNext: Run faiss_store.py")


if __name__ == "__main__":
    process_pdfs()