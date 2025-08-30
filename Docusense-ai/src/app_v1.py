import streamlit as st
from rag_utils import *

st.set_page_config(page_title="📄 PDF Chatbot (LLaMA 2)", layout="wide")
st.title("🤖 Chat with your PDF (powered by LLaMA 2 + Ollama)")

# Upload PDF
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

# Question input
query = st.text_input("Ask a question about the PDF:")

if query and "index" in st.session_state:
    with st.spinner("🤔 Thinking with LLaMA 2..."):
        context_chunks = retrieve_similar_chunks(
            query, st.session_state.chunks, st.session_state.index
        )
        context = "\n\n".join(context_chunks)

        full_prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:"""

        answer = generate_answer_with_llama2(full_prompt)

        st.markdown("### 🧠 Answer")
        st.write(answer)
