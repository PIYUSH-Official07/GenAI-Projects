import os

# 🔹 Folder name
project_dir = "pdf-chatbot-llama2"
os.makedirs(project_dir, exist_ok=True)

# 🔹 File content for requirements.txt
requirements = """streamlit
PyPDF2
faiss-cpu
sentence-transformers
requests
"""

# 🔹 File content for rag_utils.py
rag_utils = '''from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

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
'''

# 🔹 File content for app.py
app_py = '''import streamlit as st
from rag_utils import *

st.set_page_config(page_title="📄 PDF Chatbot (LLaMA 2)", layout="wide")
st.title("🤖 Chat with your PDF (powered by LLaMA 2 + Ollama)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        text = load_pdf_text(uploaded_file)
        chunks = split_text(text)
        embeddings = get_embeddings(chunks)
        index = build_vector_index(np.array(embeddings))

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.success("✅ PDF loaded and indexed!")

query = st.text_input("Ask a question about the PDF:")

if query and "index" in st.session_state:
    with st.spinner("🤔 Thinking with LLaMA 2..."):
        context_chunks = retrieve_similar_chunks(
            query, st.session_state.chunks, st.session_state.index
        )
        context = "\\n\\n".join(context_chunks)

        full_prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:"""

        answer = generate_answer_with_llama2(full_prompt)

        st.markdown("### 🧠 Answer")
        st.write(answer)
'''

# 🔹 Write all files
with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
    f.write(requirements)

with open(os.path.join(project_dir, "rag_utils.py"), "w") as f:
    f.write(rag_utils)

with open(os.path.join(project_dir, "app.py"), "w") as f:
    f.write(app_py)

print(f"✅ Project setup complete in ./{project_dir}/")
