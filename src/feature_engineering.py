import string
import re
from collections import Counter
import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import OneHotEncoder

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize vader
sia = SentimentIntensityAnalyzer()
vader = sia

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
POS_CATEGORIES = ['NN', 'NNS', 'NNP', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'RB', 'RBR', 'IN', 'DT', 'PRP']
HEDGING_PHRASES = ["i think", "maybe", "probably", "perhaps", "might", "could", "seems to", "appears to", "i feel like", "i believe"]
FIRST_PERSON = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]

def vocab_features(text):
    words = [w.lower() for w in text.split() if w.isalpha()]
    if len(words) == 0:
        return 0.0, 0.0
    freq = Counter(words)
    ttr = len(freq) / len(words)
    hapax = sum(1 for w, c in freq.items() if c == 1) / len(words)
    return ttr, hapax

def stopword_ratio(text):
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return sum(1 for w in words if w in STOPWORDS) / len(words)

def punctuation_density(text):
    if len(text) == 0:
        return 0.0
    return sum(1 for c in text if c in string.punctuation) / len(text)

def special_punct_counts(text):
    excl = text.count('!')
    quest = text.count('?')
    ellipsis = text.count('...')
    emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]', text))
    return excl, quest, ellipsis, emoji_count

def pos_distribution(text):
    try:
        words = word_tokenize(text[:2000])  # Truncate for speed
        tags = pos_tag(words)
        total = len(tags)
        if total == 0:
            return {f'pos_{t}': 0.0 for t in POS_CATEGORIES}
        counts = Counter(t for _, t in tags)
        return {f'pos_{t}': counts.get(t, 0) / total for t in POS_CATEGORIES}
    except Exception:
        return {f'pos_{t}': 0.0 for t in POS_CATEGORIES}

def ngram_repetition_rate(text, n=3):
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    freq = Counter(ngrams)
    repeated = sum(1 for ng, c in freq.items() if c > 1)
    return repeated / len(freq)

def hedging_score(text):
    text_lower = text.lower()
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    count = sum(text_lower.count(phrase) for phrase in HEDGING_PHRASES)
    return count / word_count * 100

def self_reference_rate(text):
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return sum(1 for w in words if w in FIRST_PERSON) / len(words)

def formality_score(row):
    formal = row.get('pos_NN', 0) + row.get('pos_NNS', 0) + row.get('pos_NNP', 0) + \
             row.get('pos_JJ', 0) + row.get('pos_JJR', 0) + row.get('pos_JJS', 0) + \
             row.get('pos_IN', 0) + row.get('pos_DT', 0)
    informal = row.get('pos_VB', 0) + row.get('pos_VBD', 0) + row.get('pos_VBG', 0) + \
               row.get('pos_RB', 0) + row.get('pos_RBR', 0) + row.get('pos_PRP', 0)
    total = formal + informal
    if total == 0:
        return 0.5
    return formal / total

def compute_perplexity(text, ppl_tokenizer, ppl_model, device, max_length=512):
    try:
        encodings = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        if input_ids.shape[1] < 2:
            return np.nan
        with torch.no_grad():
            outputs = ppl_model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()
    except Exception:
        return np.nan

def burstiness(text):
    sents = sent_tokenize(text)
    if len(sents) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sents]
    mean_len = np.mean(lengths)
    if mean_len == 0:
        return 0.0
    return np.std(lengths) / mean_len

def sentiment_features(text):
    scores = vader.polarity_scores(text)
    sents = sent_tokenize(text)
    if len(sents) >= 2:
        sent_compounds = [vader.polarity_scores(s)['compound'] for s in sents]
        variability = np.std(sent_compounds)
    else:
        variability = 0.0
    return scores['compound'], scores['pos'], scores['neg'], scores['neu'], variability

def get_vader_compound(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0.0
    return sia.polarity_scores(str(text))['compound']

def engineer_text_features(df, text_col='content'):
    print("Calculating text features (vocab, stopwords, burstiness, etc.)...")
    if 'to_pandas' in dir(df):
        df = df.to_pandas()
    df_out = df.copy()
    
    # Safely fill NaNs
    texts = df_out[text_col].fillna("").astype(str)
    
    # Vocab features returns (ttr, hapax)
    vocab_res = texts.apply(vocab_features)
    df_out['ttr'] = [x[0] for x in vocab_res]
    df_out['hapax'] = [x[1] for x in vocab_res]
    
    df_out['stopword_ratio'] = texts.apply(stopword_ratio)
    df_out['burstiness'] = texts.apply(burstiness)
    df_out['punctuation_density'] = texts.apply(punctuation_density)
    df_out['hedging_score'] = texts.apply(hedging_score)
    df_out['self_reference_rate'] = texts.apply(self_reference_rate)
    
    # One-Hot Encode Forum Category (if present)
    if 'subreddit' in df_out.columns and 'forum' not in df_out.columns:
        df_out.rename(columns={'subreddit': 'forum'}, inplace=True)
        
    if 'forum' in df_out.columns:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_array = encoder.fit_transform(df_out[['forum']])
        feature_names = encoder.get_feature_names_out(['forum'])
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df_out.index)
        df_out = pd.concat([df_out, encoded_df], axis=1)
    
    return df_out
