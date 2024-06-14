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

def handle_file_upload(file, tfidf, model):
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

# Function to render the home section
def render_home(tfidf, model):
    st.title("Hotel Reviews Classifier")
    st.write("This app predicts whether a hotel review is positive or negative.")
    st.markdown("#### Example Reviews")
    examples = [
        "The hotel was amazing with excellent service.",
        "Terrible experience. The room was dirty and the staff was rude.",
        "Average stay. The location was convenient but the food was just okay."
    ]
    selected_example = st.selectbox("Select an Example Review", examples)
    if st.button('Use Selected Example'):
        input_review = selected_example

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

# Function to render the upload reviews section
def render_upload_reviews(tfidf, model):
    st.title("Upload Hotel Reviews for Batch Processing")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if st.button('Process File'):
        handle_file_upload(uploaded_file, tfidf, model)

# Function to render the recent predictions section
def render_recent_predictions():
    st.title("Recent Predictions")
    if len(st.session_state['recent_predictions']) > 0:
        recent_df = pd.DataFrame(st.session_state['recent_predictions'])
        st.table(recent_df)
    else:
        st.write("No predictions made yet.")

# Function to render the feedback section
def render_feedback():
    st.title("Feedback")
    feedback = st.text_area("Please provide your feedback on the prediction accuracy or any suggestions you have.")
    if st.button('Submit Feedback'):
        if feedback.strip():
            st.success("Thank you for your feedback!")
            # Handle feedback storage or processing here (e.g., save to a database)
        else:
            st.warning("Please enter feedback before submitting.")

# Function to render the about section
def render_about():
    st.title("About")
    st.write("This application predicts whether a hotel review is positive or negative.")
    st.write("It uses a machine learning model trained on hotel review data.")

# Main function to run the app
def main():
    st.set_page_config(page_title="Hotel Reviews Classifier", page_icon=":hotel:", layout="wide")
    tfidf, model = load_model_and_vectorizer()
    init_session_state()
    authenticate_user()

    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to", ["Home", "Upload Reviews", "Recent Predictions", "Feedback", "About"])

    if menu == "Home":
        render_home(tfidf, model)
    elif menu == "Upload Reviews":
        render_upload_reviews(tfidf, model)
    elif menu == "Recent Predictions":
        render_recent_predictions()
    elif menu == "Feedback":
        render_feedback()
    elif menu == "About":
        render_about()

if __name__ == "__main__":
    main()
