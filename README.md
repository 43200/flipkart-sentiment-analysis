
# Sentiment Analysis of Flipkart Product Reviews

## 📌 Project Overview
This project focuses on performing **sentiment analysis** on real-time Flipkart product reviews to classify them as **positive** or **negative**. The goal is to understand customer satisfaction and identify common pain points from negative reviews using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The dataset used contains **8,518 reviews** of the product **YONEX MAVIS 350 Nylon Shuttle**, including review text, ratings, and other metadata.

---

## 🎯 Objectives
- Classify customer reviews into **positive** and **negative** sentiment
- Analyze negative reviews to identify common customer issues
- Build an end-to-end NLP pipeline
- Deploy the trained model as a **web application**

---

## 📂 Dataset Description
The dataset includes the following columns:
- Reviewer Name
- Review Title
- Review Text
- Ratings
- Place of Review
- Month
- Up Votes
- Down Votes

### Sentiment Labeling Logic
- Rating ≥ 4 → **Positive**
- Rating ≤ 2 → **Negative**
- Rating = 3 → Removed (neutral)

---

## 🧹 Data Preprocessing
The following preprocessing steps were applied:
- Lowercasing text
- Removing special characters and numbers
- Stopword removal
- Lemmatization using NLTK

---

## 🔢 Feature Extraction
- **TF-IDF Vectorization** (Unigrams + Bigrams)
- Maximum features limited to 5000 for efficiency

---

## 🤖 Model Training
- Model Used: **Logistic Regression**
- Train-Test Split: 80% training, 20% testing
- Class imbalance handled using stratified sampling

---

## 📊 Model Evaluation
The model was evaluated using **F1-Score**, which is suitable for imbalanced datasets.

### Performance Summary:
- **F1 Score:** ~0.95
- Strong performance on positive sentiment
- Reasonable detection of negative reviews

---

## 🔍 Insights from Negative Reviews
Common customer pain points identified include:
- Poor product quality
- Damaged shuttlecocks
- Packaging issues
- Durability concerns
- Overpricing

---

## 🌐 Web Application
A **Streamlit** web application was developed that:
- Accepts user input as a review
- Predicts sentiment in real time
- Displays results as Positive or Negative

---

## 🚀 Deployment
The application was deployed using **Streamlit Community Cloud**, providing a publicly accessible web interface.

> Streamlit Cloud was used as a deployment alternative to AWS EC2 due to billing verification restrictions, while maintaining equivalent functionality and accessibility.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Streamlit
- Git & GitHub

---

## 📁 Project Structure

flipkart-sentiment-analysis/
│
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
