import streamlit as st
import pickle
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import os
import base64

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
@st.cache(allow_output_mutation=True)
def load_model_and_vectorizer():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the correct directory.")
        st.stop()

# Initialize session state for recent predictions
def init_session_state():
    if 'recent_predictions' not in st.session_state:
        st.session_state['recent_predictions'] = []

# Function to handle user authentication
def authenticate_user():
    login_form = st.sidebar.form(key='login_form')
    username = login_form.text_input('Username')
    password = login_form.text_input('Password', type='password')
    submit_button = login_form.form_submit_button('Login')
    
    if submit_button:
        if username == "admin" and password == "password":
            st.session_state['authenticated'] = True
        else:
            st.sidebar.error("Invalid username or password")

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

# Function to handle file upload and prediction
def handle_file_upload(file, tfidf
