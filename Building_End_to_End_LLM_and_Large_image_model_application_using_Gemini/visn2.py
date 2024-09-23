import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import google.generativeai as genai
import io

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API Key not found! Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Image Demo")
st.header("Gemini Application")

# Text input
input_text = st.text_input("Input Prompt:", key="input")

# Image file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image_bytes = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes = image_bytes.getvalue()

# Button to submit
submit = st.button("Tell me about the image")

# Function to get response from Gemini API
def get_gemini_response(input_text, image_data):
    model = genai.GenerativeModel('gemini-pro-vision')
    try:
        # Construct the input list based on the available data
        inputs = []
        if input_text:
            inputs.append(input_text)
        if image_data:
            inputs.append(image_data)

        # Ensure there is valid input to pass to the API
        if not inputs:
            return "Please provide text input or upload an image."

        # Make the API call
        response = model.generate_content(inputs)
        return response.text
    except genai.errors.APIError as api_error:
        if api_error.status_code == 403:
            return "Access forbidden: Check your API key and permissions."
        else:
            return f"API Error: {api_error}"

# Handling the submit action
if submit:
    # Check if there is valid input
    if not input_text and not image_bytes:
        st.error("Please provide input text or upload an image.")
    else:
        response = get_gemini_response(input_text, image_bytes)
        st.subheader("The Response is:")
        st.write(response)
