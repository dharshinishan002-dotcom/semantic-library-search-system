import sqlite3

conn = sqlite3.connect("books.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_name TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    book_id INTEGER,
    page INTEGER,
    chunk_text TEXT,
    FOREIGN KEY(book_id) REFERENCES books(book_id)
)
""")

conn.commit()
conn.close()

print("Database created successfully")