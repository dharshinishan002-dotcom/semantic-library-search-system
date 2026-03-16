import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
FAISS_INDEX_FILE = "faiss_index.index"
METADATA_FILE = "faiss_metadata.json"
TOP_K = 5
MIN_SCORE = 0.50  # Results below this threshold are not shown
PREVIEW_LENGTH = 400  # Characters to show per result


def load_resources():
    """Load FAISS index, metadata, and embedding model once at startup."""

    print(f"Loading FAISS index   : {FAISS_INDEX_FILE}")
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
    except Exception:
        print(f"ERROR: '{FAISS_INDEX_FILE}' not found. Run faiss_store.py first.")
        exit(1)
    print(f"  ✓ Vectors in index  : {index.ntotal}")

    print(f"Loading metadata      : {METADATA_FILE}")
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: '{METADATA_FILE}' not found. Run faiss_store.py first.")
        exit(1)
    print(f"  ✓ Metadata records  : {len(metadata)}")

    print(f"Loading model         : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✓ Model ready\n")

    return index, metadata, model


def search_query(user_query: str, index: faiss.Index, metadata: list,
                 model: SentenceTransformer, top_k: int = TOP_K) -> list:
    """
    Embed the query and search FAISS for the top-k most similar chunks.

    Filters:
      - Skips results with score < MIN_SCORE (irrelevant matches)
      - Deduplicates identical chunk texts
    """
    # Embed the query with the same normalisation used during indexing
    query_vec = model.encode([user_query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    # Fetch more candidates than needed so filtering doesn't leave us short
    scores, indices = index.search(query_vec, top_k * 3)

    results = []
    seen_chunks = set()

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1 or idx >= len(metadata):
            continue

        # ── Score threshold: discard low-confidence matches ─────────────────
        if score < MIN_SCORE:
            continue

        chunk_text = metadata[idx]["chunk_text"]

        # ── Deduplicate near-identical chunks ────────────────────────────────
        # Use first 80 chars as a fingerprint to catch overlapping duplicates
        fingerprint = chunk_text[:80]
        if fingerprint in seen_chunks:
            continue
        seen_chunks.add(fingerprint)

        results.append({
            "pdf_name": metadata[idx]["pdf_name"],
            "page_number": metadata[idx]["page_number"],
            "chunk_text": chunk_text,
            "score": float(score)
        })

        if len(results) >= top_k:
            break

    return results


def print_results(results: list, query: str):
    """Print search results in a clean, readable format."""

    print(f"\nQuery: \"{query}\"")
    print("═" * 65)

    if not results:
        print("  No relevant results found.")
        print(f"  (All results scored below the relevance threshold of {MIN_SCORE})")
        print("  Try rephrasing your query with more specific terms.")
        return

    for i, r in enumerate(results, 1):
        preview = r["chunk_text"][:PREVIEW_LENGTH]
        if len(r["chunk_text"]) > PREVIEW_LENGTH:
            preview += "..."

        print(f"\n  Result {i}")
        print(f"  PDF Name  :  {r['pdf_name']}")
        print(f"  Page No   :  {r['page_number']}")
        print(
            f"  Score     :  {r['score']:.4f}  {'★ High relevance' if r['score'] > 0.70 else '✓ Good match' if r['score'] > 0.60 else ''}")
        print(f"  Text      :  {preview}")
        print("─" * 65)


def run_search_loop(index: faiss.Index, metadata: list, model: SentenceTransformer):
    """Interactive terminal query loop. Type 'exit' to quit."""

    print("=" * 65)
    print("   PDF Semantic Search  |  100 Research Papers Indexed")
    print("   Ask anything about the paper content.")
    print("   Type 'exit' to quit.")
    print("=" * 65)

    while True:
        try:
            query = input("\nEnter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if query.lower() == "exit":
            print("Exiting...")
            break

        if not query:
            print("  Please enter a question.")
            continue

        results = search_query(query, index, metadata, model)
        print_results(results, query)


if __name__ == "__main__":
    index, metadata, model = load_resources()
    run_search_loop(index, metadata, model)
