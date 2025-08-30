from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Load embedding model (small + fast)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def get_embeddings(chunks):
    return embedder.encode(chunks)

def build_vector_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def retrieve_similar_chunks(query, chunks, index, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_answer_with_llama2(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"⚠️ LLaMA 2 error: {response.text}"
    except Exception as e:
        return f"❌ LLaMA 2 API connection error: {str(e)}"
