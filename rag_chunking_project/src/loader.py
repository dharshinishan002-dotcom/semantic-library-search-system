import os
from pypdf import PdfReader

DATA_FOLDER = "rag_chunking_project/data/clean_research_papers"


def load_all_pdfs():

    documents = []

    files = os.listdir(DATA_FOLDER)

    for file in files:

        if file.endswith(".pdf"):

            print("Loading:", file)

            path = os.path.join(DATA_FOLDER, file)

            reader = PdfReader(path)

            for page_number, page in enumerate(reader.pages):

                text = page.extract_text()

                if text:
                    documents.append({
                        "book_name": file,
                        "page": page_number + 1,
                        "text": text
                    })

    return documents


if __name__ == "__main__":

    docs = load_all_pdfs()

    print("Total pages loaded:", len(docs))

    print("\nSample output:\n")

    print(docs[0])