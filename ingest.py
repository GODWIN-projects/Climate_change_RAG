import os
import pdfplumber
import hashlib
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

DATA_DIR = "data"
PERSIST_DIR = "chroma_storage"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def compute_file_hash(path):
    """Compute an MD5 hash of the file content."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_documents():
    documents = []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            continue
        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif filename.lower().endswith((".txt", ".md")):
            with open(path, encoding="utf-8") as f:
                text = f.read()
        else:
            continue
        doc_hash = compute_file_hash(path)
        documents.append((filename, text, doc_hash))
    return documents

def create_vector_store():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name="climate_change")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    docs = load_documents()

    for filename, text, doc_hash in tqdm(docs, desc="Indexing documents"):
        # Check if hash already exists in the DB
        existing = collection.get(where={"hash": doc_hash})
        if existing["ids"]:
            print(f"⚠️ Skipping already ingested file: {filename}")
            continue

        chunks = splitter.split_text(text)
        metadatas = [{"source": filename, "chunk": i, "hash": doc_hash} for i in range(len(chunks))]
        ids = [f"{filename}-{doc_hash[:6]}-{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.embed_documents(chunks)
        )

    print("✅ Document ingestion complete.")

if __name__ == "__main__":
    os.makedirs(PERSIST_DIR, exist_ok=True)
    create_vector_store()
