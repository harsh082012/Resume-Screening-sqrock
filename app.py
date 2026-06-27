import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
nltk.download("wordnet")

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screening System")
st.write("Upload your dataset and match job descriptions with suitable roles.")

@st.cache_data
def load_data():
    return pd.read_csv("resume_data.csv")

df = load_data()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

if "cleaned_resume" not in df.columns:
    resume_col = "resume" if "resume" in df.columns else df.columns[0]
    df["cleaned_resume"] = df[resume_col].astype(str).apply(clean_text)

tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(df["cleaned_resume"])

st.sidebar.header("Job Description")
job_description = st.sidebar.text_area("Paste Job Description")

if st.sidebar.button("Find Matches"):
    if job_description.strip():
        cleaned_jd = clean_text(job_description)
        jd_vector = tfidf.transform([cleaned_jd])
        scores = cosine_similarity(jd_vector, X).flatten()
        df["match_percent"] = scores * 100

        ranked = (
            df[["job_position_name", "match_percent"]]
            .sort_values(by="match_percent", ascending=False)
            .reset_index(drop=True)
        )
        ranked.index += 1

        st.success(
            f"Top Match: {ranked.iloc[0]['job_position_name']} "
            f"({ranked.iloc[0]['match_percent']:.2f}%)"
        )

        st.subheader("Top 10 Matching Positions")
        st.dataframe(ranked.head(10), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        top10 = ranked.head(10)
        ax.bar(top10.index.astype(str), top10["match_percent"])
        ax.set_xlabel("Rank")
        ax.set_ylabel("Match %")
        st.pyplot(fig)
    else:
        st.warning("Please enter a job description.")
