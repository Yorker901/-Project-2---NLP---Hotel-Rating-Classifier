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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Hotel Reviews Prediction")

input_review = st.text_area("Please enter your review here")

if st.button('Predict'):

    # 1. preprocess
    transformed_review = transform_text(input_review)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_review])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Yayy!! Positive Review")
    else:
        st.header("Negative Review :(")