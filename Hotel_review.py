# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 01:10:14 2023

@author: admin
"""

import streamlit as st
import pickle
import string
import numpy as np
import pandas as pd
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import os

ps = PorterStemmer()

# Download NLTK data files if not already present
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
try:
    nltk.word_tokenize('test')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# Function to preprocess text
def transform_text(text):
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
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the correct directory.")
    st.stop()

# Initialize session state for recent predictions
if 'recent_predictions' not in st.session_state:
    st.session_state['recent_predictions'] = []

# CSS for custom styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

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

# Function to handle file upload
def handle_file_upload(file):
    if file is not None:
        df = pd.read_csv(file)
        df['transformed'] = df['review'].apply(transform_text)
        vector_input = tfidf.transform(df['transformed'])
        df['prediction'] = model.predict(vector_input)
        df['confidence'] = model.predict_proba(vector_input).max(axis=1)
        st.write(df)
        
        # Add to recent predictions
        for _, row in df.iterrows():
            st.session_state['recent_predictions'].append({
                'review': row['review'],
                'sentiment': 'Positive' if row['prediction'] == 1 else 'Negative',
                'confidence': row['confidence'] * 100
            })
        
        # Allow download of results
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Upload Reviews", "Recent Predictions", "Feedback", "About"])

# User authentication
if not st.session_state['authenticated']:
    authenticate_user()
else:
    if menu == "Home":
        st.title("Hotel Reviews Classifier")

        st.write("This app predicts whether a hotel review is positive or negative. Please enter your review in the text box below and click 'Predict'.")

        # Example reviews
        st.markdown("#### Example Reviews")
        examples = ["The hotel was amazing with excellent service.", 
                    "Terrible experience. The room was dirty and the staff was rude.", 
                    "Average stay. The location was convenient but the food was just okay."]
        for example in examples:
            if st.button(f"Use Example: {example}"):
                input_review = example

        input_review = st.text_area("Please enter your review here", height=200, max_chars=500)

        if st.button('Predict'):
            if input_review.strip() == "":
                st.warning("Please enter a review before predicting.")
            else:
                with st.spinner('Analyzing your review...'):
                    transformed_review = transform_text(input_review)
                    vector_input = tfidf.transform([transformed_review])
                    prediction = model.predict(vector_input)
                    probability = model.predict_proba(vector_input)
                    
                    # Update recent predictions
                    st.session_state['recent_predictions'].append({
                        'review': input_review,
                        'sentiment': 'Positive' if prediction[0] == 1 else 'Negative',
                        'confidence': probability[0][prediction[0]] * 100
                    })
                    
                    # Display result
                    if prediction[0] == 1:
                        st.success(f"This looks like a Positive Review with {probability[0][1]*100:.2f}% confidence.")
                    else:
                        st.error(f"This looks like a Negative Review with {probability[0][0]*100:.2f}% confidence.")
                    
                    # Sentiment gauge chart
                    sentiment_score = probability[0][1] * 100 if prediction[0] == 1 else probability[0][0] * 100
                    gauge_chart = alt.Chart(pd.DataFrame({'Sentiment': ['Positive' if prediction[0] == 1 else 'Negative'], 'Score': [sentiment_score]})).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="Score", type="quantitative"),
                        color=alt.Color(field="Sentiment", type="nominal"),
                        tooltip=['Sentiment', 'Score']
                    ).properties(
                        title='Sentiment Score'
                    )
                    st.altair_chart(gauge_chart, use_container_width=True)
        
        # Word Cloud Section
        st.markdown("### Word Cloud of Reviews")
        wordcloud_text = ' '.join([transform_text(review['review']) for review in st.session_state['recent_predictions']])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    elif menu == "Upload Reviews":
        st.title("Upload Hotel Reviews for Batch Processing")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        handle_file_upload(uploaded_file)

    elif menu == "Recent Predictions":
        st.title("Recent Predictions")
        if len(st.session_state['recent_predictions']) > 0:
            recent_df = pd.DataFrame(st.session_state['recent_predictions'])
            st.table(recent_df)
        else:
            st.write("No predictions made yet.")

    elif menu == "Feedback":
        st.title("Feedback")
        feedback = st.text_area("Please provide your feedback on the prediction accuracy or any suggestions you have.")
        if st.button('Submit Feedback'):
            if feedback.strip():
                st.success("Thank you for your feedback!")
                # Handle feedback storage or processing here (e.g., save to a database or send via email)
            else:
                st.warning("Please enter feedback before submitting.")

    elif menu == "About":
        st.title("About")
        st.markdown("This application uses a machine learning model to classify hotel reviews into positive or negative categories.")
        st.markdown("#### Model Information")
        st.markdown("The model was trained on a dataset of hotel reviews. It uses a TF-IDF vectorizer to transform text data into numerical form and a machine learning model to make predictions. The model has an accuracy of XX% on the test data.")

# Custom CSS styling (style.css)
'''
body {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
}

.stButton>button:hover {
    background-color: #45a049;
}

.stTextArea>textarea {
    border: 2px solid #4CAF50;
}

.stTextArea>textarea:focus {
    border-color: #45a049;
}
'''
