import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


file_path = "resumes.csv"  
df = pd.read_csv(file_path)


df = df.dropna(subset=["Resume_str", "Category"])  


def clean_text(text):
    text = text.lower()  
    text = re.sub(r"\d+", "", text)  
    text = text.translate(str.maketrans("", "", string.punctuation))  
    text = " ".join([word for word in text.split() if word not in stop_words])  
    return text


df["cleaned_resume"] = df["Resume_str"].apply(clean_text)


vectorizer = TfidfVectorizer(max_features=5000)  
X = vectorizer.fit_transform(df["cleaned_resume"])
y = df["Category"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


pickle.dump(model, open("resume_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
