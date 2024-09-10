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
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

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

# Streamlit UI
st.title("GuestPulse")

st.write("This app predicts whether a hotel review is positive or negative. Please enter your review in the text box below and click 'Predict'.")

input_review = st.text_area("Please enter your review here", height=200, max_chars=500)

if st.button('Predict'):
    if input_review.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
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

# Recent predictions section
st.markdown("### Recent Predictions")
if len(st.session_state['recent_predictions']) > 0:
    recent_df = pd.DataFrame(st.session_state['recent_predictions'])
    st.table(recent_df)
else:
    st.write("No predictions made yet.")

# Clear button
if st.button('Clear'):
    input_review = ""

# Feedback section
st.markdown("---")
st.markdown("#### Feedback")
feedback = st.text_area("Please provide your feedback on the prediction accuracy or any suggestions you have.")
if st.button('Submit Feedback'):
    if feedback.strip():
        st.success("Thank you for your feedback!")
        # Handle feedback storage or processing here (e.g., save to a database or send via email)
    else:
        st.warning("Please enter feedback before submitting.")

# About section
st.markdown("---")
st.markdown("#### About")
st.markdown("This application uses a machine learning model to classify hotel reviews into positive or negative categories.")
