import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_STORE_FILE = "faiss_store.pkl"  # single combined file
TOP_K = 5  # number of results to return
MIN_SCORE = 0.68  # discard results below this similarity
PREVIEW_CHARS = 400  # how many characters of chunk text to show


# ══════════════════════════════════════════════════════════════
#  LOAD RESOURCES
# ══════════════════════════════════════════════════════════════

def load_resources():
    """
    Load FAISS index, metadata, and embedding model into memory.
    Called once at startup — keeps everything ready for fast queries.

    Returns:
        (faiss.Index, list of metadata dicts, SentenceTransformer model)
    """

    # ── Load combined store ─────────────────────────────────────
    if not os.path.exists(FAISS_STORE_FILE):
        print(f"ERROR: '{FAISS_STORE_FILE}' not found.")
        print("Run  python build_index.py  first.")
        exit(1)

    print(f"Loading store         : {FAISS_STORE_FILE}")
    with open(FAISS_STORE_FILE, "rb") as f:
        store = pickle.load(f)

    index = faiss.deserialize_index(
        np.frombuffer(store["index_bytes"], dtype=np.uint8))
    metadata = store["metadata"]
    print(f"  ✓ Vectors in index  : {index.ntotal}")
    print(f"  ✓ Metadata records  : {len(metadata)}")

    # ── Verify sizes match ──────────────────────────────────────
    if index.ntotal != len(metadata):
        print(f"\nWARNING: index has {index.ntotal} vectors but metadata has {len(metadata)} records.")
        print("Re-run  python build_index.py  to fix this.")
        exit(1)

    # ── Embedding model ─────────────────────────────────────────
    print(f"Loading model         : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✓ Model ready")

    print()
    return index, metadata, model


# ══════════════════════════════════════════════════════════════
#  SEARCH
# ══════════════════════════════════════════════════════════════

def search(query: str, index: faiss.Index, metadata: list,
           model: SentenceTransformer) -> list:
    """
    Embed the query and search the FAISS index.

    Steps:
      1. Convert query text → 768-dim normalised vector
      2. FAISS inner-product search → top (TOP_K × 3) candidates
      3. Filter by MIN_SCORE threshold
      4. Deduplicate using first-80-char fingerprint
      5. Return top TOP_K results

    Returns:
        List of dicts: pdf_name, page_number, chunk_text, score
    """

    # Embed query — same model + normalisation used during indexing
    query_vec = model.encode(
        [query.strip()],
        normalize_embeddings=True  # L2 normalise → dot product = cosine similarity
    )
    query_vec = np.array(query_vec, dtype=np.float32)

    # FAISS search — fetch extra candidates so filtering still gives TOP_K
    scores, indices = index.search(query_vec, TOP_K * 3)

    # ── Key fix: check the BEST score first ──────────────────────
    # If even the top result is below MIN_SCORE, the query is
    # completely unrelated to the indexed papers → return empty.
    best_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0
    if best_score < MIN_SCORE:
        return []

    results = []
    seen = set()

    for score, idx in zip(scores[0], indices[0]):

        # FAISS returns -1 for unfilled slots
        if idx == -1 or idx >= len(metadata):
            continue

        # Discard individual results below threshold
        if float(score) < MIN_SCORE:
            continue

        m = metadata[idx]

        # Deduplicate: use first 80 chars as fingerprint
        fingerprint = m["chunk_text"][:80]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        results.append({
            "pdf_name": m["pdf_name"],
            "page_number": m["page_number"],
            "chunk_text": m["chunk_text"],
            "score": round(float(score), 4)
        })

        if len(results) >= TOP_K:
            break

    return results


# ══════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════

def print_results(results: list, query: str):
    """Print search results in a clean readable format."""

    print(f"\nQuery: \"{query}\"")
    print("=" * 65)

    if not results:
        print("  No relevant results found.")
        print(f"  All results scored below the threshold of {MIN_SCORE}.")
        print("  Try rephrasing your query with more specific terms.")
        return

    for i, r in enumerate(results, 1):
        # Truncate long chunk text for display
        preview = r["chunk_text"][:PREVIEW_CHARS]
        if len(r["chunk_text"]) > PREVIEW_CHARS:
            preview += "..."

        # Score label
        if r["score"] > 0.72:
            label = "High relevance"
        elif r["score"] > 0.60:
            label = "Good match"
        else:
            label = ""

        print(f"\n  Result {i}")
        print(f"  PDF Name  :  {r['pdf_name']}")
        print(f"  Page No   :  {r['page_number']}")
        print(f"  Score     :  {r['score']:.4f}  {label}")
        print(f"  Text      :  {preview}")
        print("-" * 65)


# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════

def run():
    """
    Load resources once, then run an interactive query loop.
    Keeps asking for queries until user types 'exit'.
    """

    # Load everything into memory
    index, metadata, model = load_resources()

    print("=" * 65)
    print("  PDF Semantic Search — Terminal Mode")
    print(f"  {index.ntotal} chunks indexed across your research papers")
    print("  Type your question and press Enter.")
    print("  Type 'exit' to quit.")
    print("=" * 65)

    while True:
        try:
            query = input("\nEnter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C cleanly
            print("\n\nExiting...")
            break

        # Exit condition
        if query.lower() == "exit":
            print("Exiting...")
            break

        # Empty input
        if not query:
            print("  Please enter a valid question.")
            continue

        # Search and display
        results = search(query, index, metadata, model)
        print_results(results, query)


# ─── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    run()