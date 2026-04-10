import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------
# 1. Load your Moltbook pickle
# ----------------------------------------------------------------------
df = pd.read_pickle("../data/processed_v1_5_4_new_full.pkl")   # change to your file name
# The column containing text is assumed to be 'content'
df['content'] = df['content'].astype(str)  # ensure string

# ----------------------------------------------------------------------
# 2. Extract top keywords from Moltbook using TF‑IDF (each row = document)
# ----------------------------------------------------------------------
# You can adjust ngram_range, max_features, stop_words as needed
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    token_pattern=r'(?u)\b\w+\b',
    max_features=500,          # limit vocabulary size
    ngram_range=(1, 1)         # single words only; change to (1,2) for bigrams
)

# Fit and transform the Moltbook content
tfidf_matrix = vectorizer.fit_transform(df['content'])   # shape (n_docs, n_features)

# Get average TF‑IDF score per word across all documents
avg_tfidf = tfidf_matrix.mean(axis=0).A1   # convert to 1D array
feature_names = vectorizer.get_feature_names_out()

# Create a sorted DataFrame of words by average TF‑IDF
keyword_scores = pd.DataFrame({
    'word': feature_names,
    'avg_tfidf': avg_tfidf
}).sort_values('avg_tfidf', ascending=False)

# Take top 50 (or top 20 for feature engineering)
top50_moltbook_words = keyword_scores.head(50)['word'].tolist()
top20_moltbook_words = top50_moltbook_words[:20]   # for binary features

print("Top 20 Moltbook keywords (TF‑IDF within Moltbook):")
print(top20_moltbook_words)

# ----------------------------------------------------------------------
# 3. Binary feature engineering: has_keyword_<word>
# ----------------------------------------------------------------------
def add_keyword_features(df, text_col, keywords):
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        df[f'has_keyword_{kw}'] = df[text_col].str.contains(
            pattern, case=False, regex=True, na=False
        ).astype(int)
    return df

df_with_features = add_keyword_features(df, 'content', top20_moltbook_words)

# Preview the new binary columns
print("\nFirst 5 rows – sample keyword features:")
sample_cols = [f'has_keyword_{kw}' for kw in top20_moltbook_words[:5]]
print(df_with_features[sample_cols].head())

# # ----------------------------------------------------------------------
# # 4. Save the enriched DataFrame
# # ----------------------------------------------------------------------
# df_with_features.to_pickle("moltbook_with_keyword_features.pkl")
# print("\nSaved to moltbook_with_keyword_features.pkl")