import os
from pypdf import PdfReader

# get current script folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to data folder safely
DATA_FOLDER = os.path.join(BASE_DIR, "..", "data", "clean_research_papers")


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def load_all_pdfs():
    files = os.listdir(DATA_FOLDER)

    documents = []

    for f in files:
        path = os.path.join(DATA_FOLDER, f)

        print("Loading:", f)

        text = load_pdf(path)

        documents.append(text)

    return documents


if __name__ == "__main__":

    docs = load_all_pdfs()

    print("\nTotal documents loaded:", len(docs))

    print("\nSample text:\n")
    print(docs[0][:500])