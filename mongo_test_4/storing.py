import json
import numpy as np
import faiss

METADATA_JSON = "metadata.json"
FAISS_INDEX_FILE = "faiss_index.index"
FAISS_METADATA_FILE = "faiss_metadata.json"


def store_in_faiss():
    print(f"Loading: {METADATA_JSON} ...")

    try:
        with open(METADATA_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: '{METADATA_JSON}' not found. Run pdf_processing.py first.")
        return

    if not data:
        print("metadata.json is empty. Run pdf_processing.py first.")
        return

    print(f"  Records loaded     : {len(data)}")

    embeddings_list = []
    metadata_list = []

    for item in data:
        embeddings_list.append(item["embedding"])
        metadata_list.append({
            "pdf_id": item["pdf_id"],
            "pdf_name": item["pdf_name"],
            "page_number": item["page_number"],
            "chunk_text": item["chunk_text"]
        })

    # ── Convert to float32 numpy array ─────────────────────────────────────────
    embeddings_np = np.array(embeddings_list, dtype=np.float32)
    print(f"  Embedding shape    : {embeddings_np.shape}")  # (N, 768)

    # ── Build FAISS IndexFlatIP ─────────────────────────────────────────────────
    # IndexFlatIP = Inner Product search
    # Since embeddings are L2-normalised, inner product = cosine similarity.
    # Higher score = more semantically similar.
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    print(f"\nFAISS index built:")
    print(f"  Index type         : IndexFlatIP (cosine similarity)")
    print(f"  Vectors stored     : {index.ntotal}")
    print(f"  Embedding dim      : {dimension}")

    # ── Save FAISS index ────────────────────────────────────────────────────────
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"\n  ✓ Index saved      : {FAISS_INDEX_FILE}")

    # ── Save metadata without embeddings ───────────────────────────────────────
    with open(FAISS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=4)
    print(f"  ✓ Metadata saved   : {FAISS_METADATA_FILE}")

    print("\n" + "=" * 55)
    print("✓ FAISS store complete. Run query_search.py to search.")
    print("=" * 55)


if __name__ == "__main__":
    store_in_faiss()
