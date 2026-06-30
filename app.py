import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screening System")
st.markdown("**Sqrock IT Solutions Internship — Project 3** | Built with NLP & TF-IDF Cosine Similarity")
st.markdown("---")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

uploaded_file = st.file_uploader("Upload Resume Dataset CSV", type=["csv"])

st.subheader("📝 Enter Job Description")
job_description = st.text_area("Paste the job description here:", 
    value="python machine learning data science pandas numpy scikit-learn deep learning neural network tensorflow keras data analysis visualization matplotlib seaborn sql statistics",
    height=120)

if uploaded_file and job_description:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    st.success(f"Dataset loaded! {df.shape[0]} resumes found")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes", f"{df.shape[0]:,}")
    col2.metric("Job Positions", f"{df['job_position_name'].nunique():,}")
    col3.metric("Features Used", "4 text columns")

    # Combine text columns
    df['combined_text'] = (
        df['career_objective'].fillna('') + ' ' +
        df['skills'].fillna('') + ' ' +
        df['responsibilities.1'].fillna('') + ' ' +
        df['job_position_name'].fillna('')
    )
    df['cleaned_resume'] = df['combined_text'].apply(clean_text)

    # TF-IDF + Matching
    tfidf = TfidfVectorizer(max_features=1500)
    X = tfidf.fit_transform(df['cleaned_resume'])
    cleaned_jd = clean_text(job_description)
    jd_vector = tfidf.transform([cleaned_jd])
    similarity_scores = cosine_similarity(jd_vector, X).flatten()
    df['match_percent'] = (similarity_scores * 100).round(2)

    # Top matches
    st.subheader("🏆 Top Matching Resumes")
    ranked_df = df[['job_position_name', 'match_percent']].sort_values(
        by='match_percent', ascending=False).reset_index(drop=True)
    ranked_df.index += 1
    st.dataframe(ranked_df.head(10), use_container_width=True)

    # Bar chart
    st.subheader("📊 Top 10 Match Scores")
    top10 = ranked_df.head(10).copy()
    top10['Rank'] = top10.index
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    sns.barplot(x='Rank', y='match_percent', data=top10, palette='Blues_r', ax=ax1)
    ax1.set_title('Top 10 Resume Match Scores')
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Match Score (%)')
    plt.tight_layout()
    st.pyplot(fig1)

    # Job distribution
    st.subheader("📋 Resume Count by Job Position")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    df['job_position_name'].value_counts().head(15).plot(kind='bar', color='steelblue', ax=ax2)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig2)

    # Shortlisted
    st.subheader("✅ Shortlisted Candidates (Score > 30%)")
    shortlisted = ranked_df[ranked_df['match_percent'] > 30]
    st.success(f"{len(shortlisted)} candidates shortlisted!")
    st.dataframe(shortlisted, use_container_width=True)

else:
    st.info("👆 Upload the resume CSV and enter a job description to get started.")
    st.markdown("""
    **How to use:**
    1. Download Resume Dataset from Kaggle (Saugata Roy Arghya)
    2. Upload the CSV file above
    3. Enter or edit the job description
    4. App will rank all resumes by match score
    """)
