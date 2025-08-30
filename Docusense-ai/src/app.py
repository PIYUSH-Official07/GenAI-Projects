import streamlit as st
from rag_utils import *  # Make sure this module includes your custom functions

# --- Page Config ---
st.set_page_config(
    page_title="DocSense — Chat with Your PDFs",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title ---
st.markdown("<h1 style='text-align: center;'>📄 DocuSense</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>AI-Powered Answers from Any PDF</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Layout ---
col1, col2 = st.columns([1, 2])

# --- PDF Upload ---
with col1:
    st.markdown("### 📤 Upload Your PDF")
    uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("🔍 Extracting & Chunking Text..."):
            text = load_pdf_text(uploaded_file)
            chunks = split_text(text)
            store_chunks(chunks)
            st.success("✅ PDF successfully indexed into ChromaDB!")

# --- Query Input ---
with col2:
    st.markdown("### 💬 Ask a Question")
    query = st.text_input("What do you want to know from the PDF?", placeholder="e.g. What is the main topic?")

    if query:
        with st.spinner("🧠 Thinking with LLaMA2..."):
            context_chunks = retrieve_chunks(query)
            context = "\n\n".join(context_chunks)

            full_prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:"""

            answer = generate_answer_with_llama2(full_prompt)

        # --- Display Answer ---
        st.markdown("### 🧠 Answer")
        with st.container():
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <p style="font-size: 16px;">{answer}</p>
            </div>
            """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit, ChromaDB, and LLaMA2</p>", unsafe_allow_html=True)
