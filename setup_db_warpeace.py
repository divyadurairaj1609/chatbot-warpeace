import chromadb
from sentence_transformers import SentenceTransformer
import pdfplumber

print("=" * 50)
print("   Setting Up War and Peace DB")
print("=" * 50)

print("\nLoading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

client = chromadb.PersistentClient(path="./vector_db")

print("\nReading War and Peace PDF...")
texts = []
ids = []

with pdfplumber.open("war-peace.pdf") as pdf:
    total_pages = len(pdf.pages)
    print(f"Total pages: {total_pages}")
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and len(text.strip()) > 50:
            texts.append(text.strip())
            ids.append(f"page_{i+1}")
        if (i + 1) % 100 == 0:
            print(f"  Read {i+1}/{total_pages} pages...")

print(f"Extracted {len(texts)} pages!")

try:
    client.delete_collection("war_and_peace")
    print("Cleared existing collection")
except:
    pass

collection = client.create_collection("war_and_peace")

print("\nStoring in vector database...")
batch_size = 50
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    embeddings = model.encode(batch_texts).tolist()
    collection.add(
        ids=batch_ids,
        embeddings=embeddings,
        documents=batch_texts
    )
    print(f"  Stored {min(i+batch_size, len(texts))} of {len(texts)} pages")

print(f"\nDone! {collection.count()} pages stored!")
print("Run: python chatbot.py")