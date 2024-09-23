import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and store vector embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context".\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and get the response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main app function
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")

    # Apply custom CSS for styling
    st.markdown("""
        <style>
            .reportview-container {
                background-color: #f7f7f7;
            }
            .sidebar .sidebar-content {
                background-color: #f4f4f4;
            }
            .stTextInput>div>input {
                border-radius: 10px;
                border: 2px solid #3498db;
            }
            .stButton>button {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                border: none;
                padding: 10px;
                font-size: 16px;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: #2980b9;
            }
            .stMarkdown>p {
                color: #333;
            }
            .chat-bubble-user {
                background-color: #3498db;
                color: white;
                border-radius: 15px;
                padding: 10px;
                margin-bottom: 10px;
            }
            .chat-bubble-bot {
                background-color: #e1e1e1;
                border-radius: 15px;
                padding: 10px;
                margin-bottom: 10px;
            }
            .stHeader {
                margin-top: -20px; /* Adjust the margin as needed */
            }
        </style>
        """, unsafe_allow_html=True)

    # Header
    st.header("Chat with PDF using Gemini 💁")

    # Initialize session state for chat history and input if not set
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Display chat history
    for chat in st.session_state['chat_history']:
        st.markdown(f"<div class='chat-bubble-user'>**You**: {chat['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-bot'>**Bot**: {chat['answer']}</div>", unsafe_allow_html=True)

    # User input
    user_question = st.text_input("Ask a Question from the PDF Files", value="", placeholder="Type your question here...", key="user_question_input")

    # Handle the submission of user input
    if st.button("Chat"):
        if user_question.strip():
            response = user_input(user_question)
            st.session_state['chat_history'].append({"question": user_question, "answer": response})
            # Manually clear the input field
            st.session_state['user_input'] = ""  # Ensure session state input is cleared
            st.experimental_rerun()  # Refresh to update chat history

    # Sidebar for uploading and processing PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Done!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    main()
