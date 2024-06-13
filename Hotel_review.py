# # -*- coding: utf-8 -*-
# """
# Created on Sat Feb  4 01:10:14 2023

# @author: admin
# """

# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.stem.porter import PorterStemmer
# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()
    
#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Hotel Reviews Classifier")

# input_review = st.text_area("Please enter your review here")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_review = transform_text(input_review)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_review])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Yayy!! Positive Review")
#     else:
#         st.header("Negative Review :(")



# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 01:10:14 2023

@author: admin
"""

import streamlit as st
import pickle
import string
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

# Streamlit UI
st.title("Hotel Reviews Classifier")

st.write("This app predicts whether a hotel review is positive or negative.")
st.write("Please enter your review in the text box below and click 'Predict'.")

input_review = st.text_area("Please enter your review here", height=200, max_chars=500)

if st.button('Predict'):
    if input_review.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        transformed_review = transform_text(input_review)
        vector_input = tfidf.transform([transformed_review])
        prediction = model.predict(vector_input)
        probability = model.predict_proba(vector_input)
        
        if prediction[0] == 1:
            st.success(f"This looks like a Positive Review with {probability[0][1]*100:.2f}% confidence.")
        else:
            st.error(f"This looks like a Negative Review with {probability[0][0]*100:.2f}% confidence.")

# Clear button
if st.button('Clear'):
    input_review = ""

# About section
st.markdown("---")
st.markdown("#### About")
st.markdown("This application uses a machine learning model to classify hotel reviews into positive or negative categories.")


