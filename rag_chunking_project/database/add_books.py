import sqlite3
import os

# path where your pdf books are stored
BOOK_FOLDER = "rag_chunking_project/data/clean_research_papers"

# connect to database
conn = sqlite3.connect("books.db")
cursor = conn.cursor()

# read all pdf files
for file in os.listdir(BOOK_FOLDER):
    if file.endswith(".pdf"):
        cursor.execute(
            "INSERT INTO books (book_name) VALUES (?)",
            (file,)
        )
        print(f"Inserted book: {file}")

conn.commit()
conn.close()

print("All books registered in database.")