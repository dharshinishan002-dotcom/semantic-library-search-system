import sqlite3
import numpy as np
import faiss

from loader import load_all_pdfs
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def build_pipeline():

    print("Starting semantic pipeline...")

    docs = load_all_pdfs()

    # embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # semantic chunker
    chunker = SemanticChunker(embeddings)

    # connect SQL database
    conn = sqlite3.connect("books.db")
    cursor = conn.cursor()

    vectors = []

    chunk_id = 0

    for doc in docs:

        book_name = doc["book_name"]
        page = doc["page"]
        text = doc["text"]

        # find book_id
        cursor.execute(
            "SELECT book_id FROM books WHERE book_name=?",
            (book_name,)
        )

        result = cursor.fetchone()

        if result is None:
            continue

        book_id = result[0]

        # semantic chunking
        chunks = chunker.split_text(text)

        for chunk in chunks:

            # create embedding
            vector = embeddings.embed_query(chunk)

            vectors.append(vector)

            # store metadata in SQL
            cursor.execute(
                """
                INSERT INTO chunks (chunk_id, book_id, page, chunk_text)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, book_id, page, chunk)
            )

            chunk_id += 1

    conn.commit()
    conn.close()

    print("Chunks stored in SQL:", chunk_id)

    # convert vectors to numpy
    vectors = np.array(vectors).astype("float32")

    dimension = vectors.shape[1]

    # build FAISS index
    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    # save FAISS index
    faiss.write_index(index, "faiss_index.bin")

    print("FAISS vector database created")
    print("Total vectors stored:", len(vectors))


if __name__ == "__main__":

    build_pipeline()