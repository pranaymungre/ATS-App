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


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


model = pickle.load(open("resume_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    return text.strip()


def clean_text(text):
    text = text.lower()  
    text = re.sub(r"\d+", "", text)  
    text = text.translate(str.maketrans("", "", string.punctuation)) 
    text = " ".join([word for word in text.split() if word not in stop_words])  
    return text


def calculate_ats_score(job_description, resume_text):
    job_keywords = set(clean_text(job_description).split())  
    resume_words = set(clean_text(resume_text).split())  
    matched_keywords = job_keywords.intersection(resume_words) 
    if len(job_keywords) == 0:  
        return 0
    return round((len(matched_keywords) / len(job_keywords)) * 100, 2) 


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
            resume_text = extract_text_from_pdf(uploaded_file)  
            cleaned_resume = clean_text(resume_text)  
            vectorized_resume = vectorizer.transform([cleaned_resume])  

           
            probabilities = model.predict_proba(vectorized_resume)[0]
            predicted_category = model.classes_[np.argmax(probabilities)]
            confidence_score = round(np.max(probabilities) * 100, 2)

        
            ats_score = calculate_ats_score(job_description, resume_text)

           
            final_score = round((confidence_score * 0.6) + (ats_score * 0.4), 2)

            resume_data.append({"Resume Name": uploaded_file.name, 
                                "Predicted Category": predicted_category, 
                                "Confidence Score": confidence_score,
                                "ATS Score": ats_score,
                                "Final Score (Rank)": final_score})

       
        df_results = pd.DataFrame(resume_data)
        df_results = df_results.sort_values(by="Final Score (Rank)", ascending=False)

       
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
