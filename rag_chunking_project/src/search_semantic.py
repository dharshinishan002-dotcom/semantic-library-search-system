
import faiss
import sqlite3
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
index = faiss.read_index("faiss_index.bin")
print("FAISS vectors loaded:", index.ntotal)

# Connect SQLite database
conn = sqlite3.connect("books.db")
cursor = conn.cursor()


def search(query, k=5):

    # Convert query to embedding
    query_vector = embeddings.embed_query(query)

    # Convert to numpy format required by FAISS
    query_vector = np.array([query_vector]).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_vector, k)

    print("\nTop results:\n")

    for idx in indices[0]:

        cursor.execute(
            "SELECT book_id, page, chunk_text FROM chunks WHERE chunk_id=?",
            (int(idx),)
        )

        row = cursor.fetchone()

        if row:
            book_id, page, text = row

            print("Book ID:", book_id)
            print("Page:", page)
            print("Text:", text[:400])
            print("\n----------------------\n")


if __name__ == "__main__":

    while True:

        query = input("\nAsk question: ")

        if query.lower() in ["exit", "quit"]:
            break

        search(query)
