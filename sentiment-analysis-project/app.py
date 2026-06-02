import streamlit as st
import joblib
import re
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download stopwords (only first run)
nltk.download('stopwords')

# Load model & vectorizer
# model = joblib.load("sentiment_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

# Stopwords and stemmer
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# --- Preprocessing Functions ---
def review_cleaning(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def perform_stemming(text):
    words = text.split()
    stemmed_words = [ps.stem(word) for word in words]
    return " ".join(stemmed_words)

def preprocess_text(text):
    text = review_cleaning(text)
    text = remove_stopwords(text)
    text = perform_stemming(text)
    return text

# Sentiment labels
sentiment_classes = ["Negative", "Neutral", "Positive"]

# --- Streamlit UI ---
st.title("🎵 Musical Instrument Review Sentiment Analysis")

# Dropdown for instrument
instrument = st.selectbox(
    "Choose a musical instrument:",
    ["Guitar", "Piano", "Drums", "Violin", "Flute", "Other"]
)

# Text input for review
user_review = st.text_area("Write your review here:")

# Predict button
if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        # Preprocess input
        preprocessed_text = preprocess_text(user_review)
        input_vectorized = vectorizer.transform([preprocessed_text])

        # Predict sentiment
        prediction = model.predict(input_vectorized)[0]
        predicted_sentiment = sentiment_classes[prediction]

        # Display result
        st.success(f"🎶 Instrument: {instrument}")
        st.write(f"📝 Your Review: {user_review}")
        st.subheader(f"✅ Predicted Sentiment: **{predicted_sentiment}**")
