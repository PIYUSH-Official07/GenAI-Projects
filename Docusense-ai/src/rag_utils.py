from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import HttpClient
import requests

# Embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to ChromaDB running on localhost:8000
client = HttpClient(host="localhost", port=8000)
COLLECTION_NAME = "pdf_chunks"

def load_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text

def split_text(text, chunk_size=500, overlap=50):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def store_chunks(chunks):
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

def retrieve_chunks(query, k=3):
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results['documents'][0]

def generate_answer_with_llama2(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": "llama2", "prompt": prompt, "stream": False}
    try:
        res = requests.post(url, headers=headers, json=data)
        return res.json()["response"] if res.status_code == 200 else f"⚠️ Error: {res.text}"
    except Exception as e:
        return f"❌ API error: {str(e)}"
