from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page_num, page in enumerate(reader.pages):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"

    return text


if __name__ == "__main__":
    # Path to your PDF
    file_path = "../data/book.pdf"

    # Load PDF text
    document_text = load_pdf(file_path)

    print("PDF Loaded Successfully!")
    print("Total Characters:", len(document_text))
    print("\n--- Sample Extracted Text ---\n")
    print(document_text[:1000])  # show first 1000 characters
