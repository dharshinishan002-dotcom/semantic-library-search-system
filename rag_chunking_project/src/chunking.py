import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_all_pdfs


def recursive_chunking():

    documents = load_all_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)

    print("Total chunks created:", len(all_chunks))

    print("\nSample chunk:\n")
    print(all_chunks[0])
    return all_chunks


if __name__ == "__main__":
    recursive_chunking()