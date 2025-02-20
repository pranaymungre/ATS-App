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

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load dataset
file_path = "resumes.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path)

# Check for missing values
df = df.dropna(subset=["Resume_str", "Category"])  # Remove empty rows

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing
df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 words
X = vectorizer.fit_transform(df["cleaned_resume"])
y = df["Category"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
pickle.dump(model, open("resume_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
