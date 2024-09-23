from dotenv import load_dotenv
load_dotenv()

import base64
import streamlit as st
import os
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from constant import watsonx_project_id, api_key

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": api_key,
}

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

# Model parameters
model_id = ModelTypes.LLAMA_2_70B_CHAT
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

# Initialize the model
mt0_xxl = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=watsonx_project_id
)

def get_gemini_response(input, pdf_content, prompt):
    # Initialize WatsonxLLM
    model = WatsonxLLM(model=mt0_xxl)
    # Ensure all inputs are strings and put into a list
    inputs = [input, pdf_content[0], prompt]
    # Generate response using the model
    response = model.generate(inputs)
    return response.generations[0][0].text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        # pdf_parts = [
        #     {
        #         "mime_type": "image/jpeg",
        #         "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
        #     }
        # ]
        pdf_parts = base64.b64encode(img_byte_arr).decode()  # encode to base64
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

## Streamlit App

st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("Tell Me About the Resume")
submit2 = st.button("How Can I Improvise my Skills")
submit3 = st.button("Percentage match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager with Tech Experience in the field of Data Science, Web development, Big Data Engineering, DevOps, Data Analyst. Your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt2 = """
You are an experienced Technical Human Resource Manager with expertise in the field of Data Science, Web development, Big Data Engineering, DevOps, Data Analyst. 
Your role is to scrutinize the provided resume in light of the job description provided. 
Please share your insight on whether the candidate's profile aligns with the role. 
Additionally, offer advice on enhancing the candidate's skills and identify areas of improvement.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description. Provide the percentage of match if the resume matches
the job description. First, the output should come as a percentage, then keywords missing, and finally, your final thoughts.
"""

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt1,pdf_content,input_text)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.write("Please upload the resume")
elif submit2:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt2)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.write("Please upload the resume")
elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt3)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.write("Please upload the resume")
