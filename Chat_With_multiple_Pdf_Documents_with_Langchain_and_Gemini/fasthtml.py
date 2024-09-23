from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fasthtml.common import html, head, body, h1, form, input, button, pre, script
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from io import BytesIO
from typing import List

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(BytesIO(pdf))
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
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    pdf_text = get_pdf_text([await file.read() for file in files])
    text_chunks = get_text_chunks(pdf_text)
    get_vector_store(text_chunks)
    return JSONResponse(content={"message": "PDFs processed and vector store updated"})

@app.get("/ask/")
async def ask_question(question: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return JSONResponse(content={"answer": response["output_text"]})

@app.get("/", response_class=HTMLResponse)
async def main_page():
    html_content = html(
        head(
            meta(charset="UTF-8"),
            meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            title("Chat with PDF using Gemini")
        ),
        body(
            h1("Chat with PDF using Gemini💁"),
            form(
                id="uploadForm",
                enctype="multipart/form-data",
                action="/upload/",
                method="post",
                children=[
                    input(type="file", id="files", name="files", multiple=True, accept=".pdf"),
                    button(type="submit")("Submit & Process")
                ]
            ),
            form(
                id="questionForm",
                action="/ask/",
                method="get",
                children=[
                    input(type="text", id="question", name="question", placeholder="Ask a question from the PDF files", required=True),
                    button(type="submit")("Ask")
                ]
            ),
            h2("Response"),
            pre(id="response")(),
            script("""
                document.getElementById('uploadForm').onsubmit = async function(event) {
                    event.preventDefault();
                    const formData = new FormData(this);
                    const response = await fetch('/upload/', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    alert(result.message);
                };

                document.getElementById('questionForm').onsubmit = async function(event) {
                    event.preventDefault();
                    const question = document.getElementById('question').value;
                    const response = await fetch(`/ask/?question=${encodeURIComponent(question)}`);
                    const result = await response.json();
                    document.getElementById('response').innerText = result.answer;
                };
            """)
        )
    )
    return HTMLResponse(content=html_content)
