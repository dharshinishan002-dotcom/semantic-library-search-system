# ResearchLens — PDF Semantic Search

A full-stack semantic search system for research papers. Search across 100+ PDFs using natural language queries powered by FAISS vector search and sentence embeddings — built from scratch without any RAG frameworks.

---

## Features

- **Search** — type any research question and get the most relevant text chunks from all indexed PDFs
- **Click to open** — click any result to open the original PDF directly in the browser
- **Upload** — add new PDFs from the browser and have them indexed automatically
- **Filter** — narrow search results to a specific PDF using the sidebar
- **No results for irrelevant queries** — queries unrelated to research papers show "No results found"

---

## Project structure

```
project/
│
├── build_index.py      # Step 1 — extract, chunk, embed, build FAISS index
├── backend.py          # Step 2 — FastAPI server (all API endpoints)
├── query_search.py     # Optional — terminal-based search
├── index.html          # Step 3 — browser UI
│
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from Git
│
└── faiss_store.pkl     # Generated — combined FAISS index + metadata (not in git)
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| PDF extraction | PyMuPDF (fitz) | Read text page by page |
| Embeddings | SentenceTransformers | Convert text to 768-dim vectors |
| Embedding model | BAAI/bge-base-en-v1.5 | Semantic understanding |
| Vector search | FAISS IndexFlatIP | Cosine similarity search |
| Index storage | Python pickle | Single combined file |
| Database | MongoDB | Store PDF metadata and filepaths |
| Backend | FastAPI + Uvicorn | REST API on port 8000 |
| Frontend | HTML + JS | Browser UI on port 3000 |

> No RAG frameworks (LangChain, LlamaIndex) used. Everything built from scratch.

---

## How it works

### Indexing pipeline (build_index.py)

```
MongoDB pdf_files collection
        ↓
Read filepath for each PDF
        ↓
Extract text page by page  (PyMuPDF)
        ↓
Filter out reference pages, TOC pages, short pages
        ↓
Split into 400-word chunks with 80-word overlap
        ↓
Generate 768-dim embedding per chunk  (BAAI/bge-base-en-v1.5)
        ↓
Build FAISS IndexFlatIP
        ↓
Save combined index + metadata → faiss_store.pkl
```

### Search pipeline (backend.py + index.html)

```
User query (text)
        ↓
Embed query → 768-dim normalised vector
        ↓
FAISS index.search() — dot product against all stored vectors
        ↓
Returns top indices + scores
        ↓
metadata[idx] → pdf_name, page_number, chunk_text
        ↓
Filter score ≥ 0.68 (irrelevant queries return "No results found")
        ↓
Top 5 results returned to browser
```

### Why one combined file (faiss_store.pkl)

Previously the project saved two separate files — `faiss_index.index` (FAISS binary) and `faiss_metadata.json` (text info). These are now combined into a single `faiss_store.pkl` using Python pickle:

```python
# Saving
store = {
    "index_bytes": faiss.serialize_index(index).tobytes(),
    "metadata":    metadata_list
}
pickle.dump(store, open("faiss_store.pkl", "wb"))

# Loading
store    = pickle.load(open("faiss_store.pkl", "rb"))
index    = faiss.deserialize_index(np.frombuffer(store["index_bytes"], dtype=np.uint8))
metadata = store["metadata"]
```

---

## Prerequisites

- Python 3.10 or higher
- MongoDB running locally on port 27017
- PDFs stored on disk with their filepaths saved in MongoDB

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/researchlens.git
cd researchlens
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Make sure MongoDB is running

```bash
mongod
```

Your `research_papers` database must have a `pdf_files` collection where each document looks like:

```json
{
  "_id": "ObjectId(...)",
  "filename": "2301.08028v4.pdf",
  "filepath": "/Users/yourname/Desktop/clean_research_papers/2301.08028v4.pdf",
  "uploaded_at": "2026-03-13T13:01:58+00:00"
}
```

---

## Running the project

### Step 1 — Build the FAISS index (run once)

```bash
python build_index.py
```

Expected output:
```
✓ Model ready
Found 101 PDFs in MongoDB.
Processing : 2301.08028v4.pdf
  Pages: 164  |  Skipped ref:12 toc:3 short:8  |  Chunks: 413
...
✓ Saved combined store : faiss_store.pkl
  Total PDFs processed : 101
  Total chunks indexed : 4618
```

### Step 2 — Start the backend (Terminal 1)

```bash
uvicorn backend:app --reload --port 8000
```

Expected output:
```
✓ Model ready
✓ FAISS store loaded — 4618 vectors, 4618 records
Server ready at http://localhost:8000
```

### Step 3 — Start the frontend server (Terminal 2)

```bash
python -m http.server 3000
```

### Step 4 — Open the browser

```
http://localhost:3000
```

Click `index.html` in the file list if needed.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/stats` | Total PDFs, chunks, index status |
| `GET` | `/api/pdfs` | List all PDFs from MongoDB |
| `POST` | `/api/search` | Semantic search query |
| `POST` | `/api/upload` | Upload and auto-index a new PDF |
| `GET` | `/api/pdf/{pdf_id}` | Serve PDF file inline in browser |

### Search request example

```json
POST /api/search
{
  "query": "What is meta-reinforcement learning?",
  "top_k": 5,
  "pdf_filter": ""
}
```

### Search response example

```json
{
  "query": "What is meta-reinforcement learning?",
  "total": 3,
  "results": [
    {
      "pdf_name": "2301.08028v4.pdf",
      "page_number": 5,
      "chunk_text": "Meta-reinforcement learning refers to...",
      "score": 0.8738
    }
  ]
}
```

---

## Score threshold

The `MIN_SCORE = 0.68` in `backend.py` controls when results are shown vs hidden.

| Score | Meaning |
|---|---|
| 0.80+ | Very relevant — directly answers the query |
| 0.70–0.79 | Good match |
| 0.68–0.69 | Borderline — just above threshold |
| Below 0.68 | Irrelevant — returns "No results found" |

To adjust the threshold, change this line in `backend.py`:
```python
MIN_SCORE = 0.68    # increase to be stricter, decrease to be more lenient
```

---

## Chunking strategy

| Parameter | Value | Reason |
|---|---|---|
| Chunk size | 400 words | Large enough for meaningful context |
| Overlap | 80 words | Prevents losing context at boundaries |
| Min page words | 50 words | Skips nearly empty pages |

### Pages automatically filtered out

- **Reference pages** — 55%+ of lines contain citation patterns (URLs, arXiv IDs, "et al.")
- **Table of contents pages** — 40%+ of lines contain dot sequences `......`
- **Short pages** — fewer than 50 words

---

## Terminal search (optional)

You can also search directly from the terminal without starting the backend:

```bash
python query_search.py
```

```
Loading store         : faiss_store.pkl
  ✓ Vectors in index  : 4618
  ✓ Metadata records  : 4618
  ✓ Model ready

PDF Semantic Search — Terminal Mode
Type your question and press Enter.
Type 'exit' to quit.

Enter your question: What is meta-reinforcement learning?

  Result 1
  PDF Name  :  2301.08028v4.pdf
  Page No   :  5
  Score     :  0.8738  High relevance
  Text      :  Meta-reinforcement learning refers to...
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not import module "main"` | Use `uvicorn backend:app` not `uvicorn main:app` |
| `[Errno 48] Address already in use` | Run `lsof -ti :8000 | xargs kill -9` then restart |
| `Failed to fetch` in browser | Make sure backend is running on port 8000 |
| `python-multipart` error | Run `pip install python-multipart` |
| `faiss_store.pkl not found` | Run `python build_index.py` first |
| `ValueError: document closed` | Re-download `build_index.py` — bug was fixed |
| Irrelevant results showing | Increase `MIN_SCORE` in `backend.py` |
| Too many "No results found" | Decrease `MIN_SCORE` in `backend.py` |

---

## Regenerating the index

If you add PDFs manually to MongoDB (not via upload), delete the old index and rebuild:

```bash
rm faiss_store.pkl
python build_index.py
```

Then restart the backend server.

---

## License

MIT License — free to use and modify.