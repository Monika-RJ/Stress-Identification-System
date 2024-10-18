import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved models and vectorizers
with open('svm_classifier.pkl', 'rb') as svm_file:
    svm_classifier = pickle.load(svm_file)

with open('tfidf_vectorizer (3).pkl', 'rb') as svm_vectorizer_file:
    svm_vectorizer = pickle.load(svm_vectorizer_file)

with open('random_forest.pkl', 'rb') as rf_file:
    rf_classifier = pickle.load(rf_file)

with open('tfidf_vectorizer2.pkl', 'rb') as rf_vectorizer_file:
    rf_vectorizer = pickle.load(rf_vectorizer_file)

# Streamlit app configuration
st.title("Stress Identification System")
st.write("This app classifies text as either 'Stressed' or 'Non stressed'.")

# Input box for user to enter the text
user_input = st.text_area("Enter the text you want to classify:")

# Dropdown to select model (SVM or Random Forest)
model_choice = st.selectbox("Choose the classification model:", ("Tamil", "Telugu"))

# Prediction logic based on selected model
if st.button("Classify"):
    if user_input:
        if model_choice == "Tamil":
            # Preprocess and predict using SVM
            user_input_tfidf = svm_vectorizer.transform([user_input])
            prediction = svm_classifier.predict(user_input_tfidf)
        else:
            # Preprocess and predict using Random Forest
            user_input_tfidf = rf_vectorizer.transform([user_input])
            prediction = rf_classifier.predict(user_input_tfidf)
        
        # Output prediction
        label = "Stressed" if prediction == 1 else "Non stressed"
        st.write(f"Prediction: {label}")
    else:
        st.write("Please enter some text to classify.")

# Run the app using: `streamlit run app.py`
