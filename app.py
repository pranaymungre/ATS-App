import streamlit as st
import pickle
import pandas as pd
import numpy as np
import fitz  
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load trained model and vectorizer
model = pickle.load(open("resume_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    return text.strip()

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to calculate ATS score
def calculate_ats_score(job_description, resume_text):
    job_keywords = set(clean_text(job_description).split())  # Extract keywords from job description
    resume_words = set(clean_text(resume_text).split())  # Extract words from resume
    matched_keywords = job_keywords.intersection(resume_words)  # Find common words
    if len(job_keywords) == 0:  # Avoid division by zero
        return 0
    return round((len(matched_keywords) / len(job_keywords)) * 100, 2)  # ATS Score as percentage

# Streamlit UI
st.set_page_config(page_title="AI Resume Screener with ATS", layout="wide")
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Upload & Rank Resumes", "Insights"])

if selection == "Home":
    st.header("Welcome to AI Resume Screener")
    st.write("Upload resumes (PDF format) and get ranked predictions instantly!")

elif selection == "Upload & Rank Resumes":
    st.header("Upload Resumes for Screening & Ranking")
    job_description = st.text_area("Enter Job Description", placeholder="Paste the job description here...")

    uploaded_files = st.file_uploader("Upload Multiple Resumes (PDF Format)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        resume_data = []
        
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(uploaded_file)  # Extract text from PDF
            cleaned_resume = clean_text(resume_text)  # Preprocess text
            vectorized_resume = vectorizer.transform([cleaned_resume])  # Convert to numerical representation

            # Get category and confidence score
            probabilities = model.predict_proba(vectorized_resume)[0]
            predicted_category = model.classes_[np.argmax(probabilities)]
            confidence_score = round(np.max(probabilities) * 100, 2)

            # Calculate ATS Score
            ats_score = calculate_ats_score(job_description, resume_text)

            # Final ranking score (combining ATS & Confidence Score)
            final_score = round((confidence_score * 0.6) + (ats_score * 0.4), 2)

            resume_data.append({"Resume Name": uploaded_file.name, 
                                "Predicted Category": predicted_category, 
                                "Confidence Score": confidence_score,
                                "ATS Score": ats_score,
                                "Final Score (Rank)": final_score})

        # Convert to DataFrame and sort by final score
        df_results = pd.DataFrame(resume_data)
        df_results = df_results.sort_values(by="Final Score (Rank)", ascending=False)

        # Display ranked resumes
        st.subheader("Ranked Resumes by Final Score")
        st.dataframe(df_results)

elif selection == "Insights":
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("Resume Category Distribution")
    df = pd.read_csv("resumes.csv")
    category_counts = df["Category"].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
