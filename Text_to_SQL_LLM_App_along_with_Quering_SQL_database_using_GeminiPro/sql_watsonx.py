from dotenv import load_dotenv
 # Load all the environment variables

from constant import watsonx_project_id, api_key

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": api_key,
}

import streamlit as st
import os
import sqlite3

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

# Model parameters
model_id = ModelTypes.MT0_XXL
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1
}

# Initialize the model
granite_chatv2 = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=watsonx_project_id
)

# Initialize WatsonxLLM
model = WatsonxLLM(model=granite_chatv2)

# Function to get Gemini response
def get_gemini_response(question, prompt):
    try:
        response = model.generate([prompt[0], question])
        # Assuming 'generated_text' holds the SQL query string
        sql_query = response.generations[0][0].text
        return sql_query
    except AttributeError as e:
        st.error(f"Error: {e}")
        return None

# Function to read SQL query
def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except sqlite3.Error as e:
        st.error(f"SQL error: {e}")
        rows = []
    finally:
        conn.commit()
        conn.close()
    return rows

# Prompt template
prompt = [
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, SECTION.
    For example:
    Example 1 - How many entries of records are present?, the SQL command will be: SELECT COUNT(*) FROM STUDENT;
    Example 2 - Tell me all the students studying in Data Science class?, the SQL command will be: SELECT * FROM STUDENT where CLASS='Data Science';
    Ensure the SQL code does not have ``` in the beginning or end and no SQL word in the output.
    """
]

# Streamlit app configuration
st.set_page_config(page_title="I can Retrieve Any SQL query")
st.header("Gemini App To Retrieve SQL Data")

# Input from user
question = st.text_input("Input: ", key="input")

# Button to submit
submit = st.button("Ask the question")

# Process the query on submit
if submit:
    sql_query = get_gemini_response(question, prompt)
    if sql_query:
        st.subheader("Generated SQL Query")
        st.code(sql_query)
        sql_result = read_sql_query(sql_query, "student.db")
        st.subheader("Query Results")
        for row in sql_result:
            st.write(row)
