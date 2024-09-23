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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Multi PDF Chat", page_icon="🤖", layout="wide")

# CSS for custom styling
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Main background and text color */
        .main {
            background: linear-gradient(to right, #d8e2dc, #ffe5d9); /* Light gradient background */
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        /* Header and Sidebar styling */
        .header, .stSidebar {
            background-color: #4A90E2; /* Same blue background for header and sidebar */
            color: white;
        }
        /* Header styling */
        .header {
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-radius: 0 0 10px 10px; /* Rounded bottom corners */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Shadow for better visibility */
        }
        .header h1 {
            margin: 0;
            font-size: 2em;
        }
        .header .hamburger {
            font-size: 24px;
            cursor: pointer;
        }
        /* Sidebar styling */
        .stSidebar {
            color: white;
            padding: 10px;
            position: relative;
        }
        .stSidebar .stFileUploader {
            background-color: #F9FAFB; /* Very light grey for file uploader section */
            border-radius: 5px;
            padding: 10px;
            border: 2px solid #1e90ff; /* Blue border for file uploader */
            margin-top: 20px; /* Ensure spacing */
        }
        .stSidebar h1 {
            color: white; /* White text for title */
            text-align: center;
            padding: 10px;
            margin-top: 0; /* Remove default margin */
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Shadow for better visibility */
        }
        /* Deploy Header Color */
        .deploy-header {
            background-color: #FF6F61; /* Example color for deploy header */
            color: white;
        }
        /* Button styling */
        .stButton button {
            background-color: #1e90ff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #1c86ee;
        }
        /* Text input styling */
        .stTextInput input {
            border: 2px solid #1e90ff;
            padding: 10px;
            border-radius: 5px;
        }
        /* Welcome message styling */
        .welcome-message {
            background-color: #f0f8ff; /* Light background for welcome message */
            color: #333; /* Dark text color */
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Shadow for better visibility */
        }
        /* Responsive styling */
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                align-items: flex-start;
            }
            .header h1 {
                font-size: 1.5em;
            }
            .stButton button {
                width: 100%;
                font-size: 14px;
            }
            .stTextInput input {
                width: 100%;
                font-size: 14px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()

# JavaScript to toggle sidebar visibility
def add_custom_js():
    st.markdown(
        """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const hamburger = document.querySelector('.hamburger');
            const sidebar = document.querySelector('.css-1v3fvcr'); // Adjust the selector as needed
            if (hamburger && sidebar) {
                hamburger.addEventListener('click', function() {
                    sidebar.classList.toggle('css-1v3fvcr'); // Toggle class to show/hide sidebar
                });
            }
        });
        </script>
        """,
        unsafe_allow_html=True
    )

add_custom_js()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.75)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.markdown('<div class="header"> <span class="hamburger">&#9776;</span> <h1>Talent Advisor 🤖</h1> </div>', unsafe_allow_html=True)

    # Welcome message
    st.markdown('<div class="welcome-message">Welcome to Talent Advisor! How can I assist you today?</div>', unsafe_allow_html=True)

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.write('<div class="stFileUploader">', unsafe_allow_html=True)  # Wrap file uploader in custom CSS class
        st.title("Talent Advisor:")
        pdf_docs = st.file_uploader("Please upload the candidate profiles in pdf format below and click on the submit and process button once done.", accept_multiple_files=True)
        st.write('</div>', unsafe_allow_html=True)  # Close custom CSS class

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
