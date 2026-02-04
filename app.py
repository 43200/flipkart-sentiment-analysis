import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function (same logic as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# App UI
st.set_page_config(page_title="Flipkart Review Sentiment Analysis")

st.title("🛍️ Flipkart Product Review Sentiment Analysis")
st.write("Enter a product review to predict whether the sentiment is **Positive** or **Negative**.")

review = st.text_area("✍️ Enter Review Text")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(review)
        review_vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vector)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
