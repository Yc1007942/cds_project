import pandas as pd
import numpy as np
import pickle

def parse_timestamp(ts):
    if pd.isna(ts):
        return pd.NaT
    ts = str(ts).strip()
    try:
        val = float(ts)
        if val > 1e12:
            val = val / 1000
        return pd.to_datetime(val, unit='s', utc=True)
    except (ValueError, OverflowError):
        pass
    try:
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return pd.NaT

def extract_hour_feature(df, time_col='created_at'):
    """
    Extracts the hour (0-23) from a datetime column and creates a new 'hour' column.
    """
    print(f"Extracting hour from '{time_col}'...")
    if 'to_pandas' in dir(df):
        df = df.to_pandas()
    df_out = df.copy()
    
    # Ensure it's a datetime column
    if not pd.api.types.is_datetime64_any_dtype(df_out[time_col]):
        df_out[time_col] = df_out[time_col].apply(parse_timestamp)
        
    df_out['hour'] = df_out[time_col].dt.hour
    
    # Fill any null hours with an arbitrary default (e.g. 0) or the mean, if necessary
    df_out['hour'] = df_out['hour'].fillna(0)
    return df_out

def get_early_comments(df_comments, post_id_col='post_id', time_col='created_at'):
    print("Extracting early comments...")
    df_comm = df_comments.copy()
    if 'to_pandas' in dir(df_comm):
        df_comm = df_comm.to_pandas()
    df_comm[time_col] = pd.to_datetime(df_comm[time_col])
    df_early = df_comm.sort_values([post_id_col, time_col])
    return df_early.groupby(post_id_col).head(10).copy()

def engineer_comment_existence(df_early_comments, post_id_col='post_id'):
    print("Calculating Comment Existence feature...")
    if 'to_pandas' in dir(df_early_comments):
        df_early_comments = df_early_comments.to_pandas()
    existence_features = df_early_comments.groupby(post_id_col).agg(
        actual_comment_count=('id', 'count')
    )
    existence_features['comment_existence'] = existence_features['actual_comment_count'] / 10.0
    return existence_features[['comment_existence']]

def engineer_early_sentiment(df_early_comments, get_vader_compound_func, post_id_col='post_id', text_col='content'):
    print("Calculating VADER Sentiment features...")
    if 'to_pandas' in dir(df_early_comments):
        df_early_comments = df_early_comments.to_pandas()
    df_early_comments['vader_score'] = df_early_comments[text_col].apply(get_vader_compound_func)
    sentiment_features = df_early_comments.groupby(post_id_col).agg(
        avg_early_sentiment=('vader_score', 'mean'),
        max_early_sentiment=('vader_score', 'max'),
        min_early_sentiment=('vader_score', 'min')
    )
    return sentiment_features

def merge_engineered_features(df_posts, existence_df, sentiment_df, post_id_col_posts='id'):
    print("Merging features into main posts dataframe...")
    df_merged = df_posts.copy()
    if 'to_pandas' in dir(df_merged):
        df_merged = df_merged.to_pandas()
    
    df_merged = df_merged.merge(
        existence_df, left_on=post_id_col_posts, right_index=True, how='left'
    )
    df_merged['comment_existence'] = df_merged['comment_existence'].fillna(0.0)

    df_merged = df_merged.merge(
        sentiment_df, left_on=post_id_col_posts, right_index=True, how='left'
    )
    fill_cols = ['avg_early_sentiment', 'max_early_sentiment', 'min_early_sentiment']
    df_merged[fill_cols] = df_merged[fill_cols].fillna(0.0)

    return df_merged

def save_final_features(df, embeddings, output_dir="../data", dataset_prefix="data", extra_features=None):
    if 'to_pandas' in dir(df):
        df = df.to_pandas()
    df_final = df.copy()
    
    if extra_features is None:
        extra_features = ['comment_existence', 'hour', 'ttr', 'hapax', 'stopword_ratio',
                          'burstiness', 'punctuation_density', 'hedging_score', 'self_reference_rate']
        
    forum_cols = [c for c in df_final.columns if c.startswith('forum_')]
    extra_features.extend(forum_cols)

    if 'split' not in df_final.columns:
        np.random.seed(42)
        df_final['split'] = np.random.choice(['train', 'val', 'test'], size=len(df_final), p=[0.7, 0.15, 0.15])

    for split_name in ['train', 'val', 'test']:
        idx = (df_final['split'] == split_name).values
        
        present_features = [f for f in extra_features if f in df_final.columns]
        
        if present_features:
            X_split = np.hstack([
                df_final.loc[idx, present_features].values,
                embeddings[idx]
            ])
        else:
            X_split = embeddings[idx]

        split_filename = f"{output_dir}/{dataset_prefix}_features_{split_name}.pkl"
        with open(split_filename, 'wb') as f:
            pickle.dump(X_split, f)
        print(f"[{split_name.upper()}] Saved features (shape: {X_split.shape}) to {split_filename}")

    return df_final

def save_everything_to_pickle(df, embeddings, output_dir="../data", dataset_prefix="data"):
    if 'to_pandas' in dir(df):
        df = df.to_pandas()
    df_pd = df.copy()
    df_pd['embeddings'] = list(embeddings)
    pickle_filename = f"{output_dir}/{dataset_prefix}_metadata_with_embeddings.pkl"
    df_pd.to_pickle(pickle_filename)
    print(f"Saved full dataframe with embeddings shape: {df_pd.shape} to {pickle_filename}")
    return df_pd

def save_unsplit_features(df, embeddings, output_dir="../data", dataset_prefix="data", extra_features=None):
    if 'to_pandas' in dir(df):
        df = df.to_pandas()
    df_pd = df.copy()
    
    if extra_features is None:
        extra_features = ['comment_existence', 'hour', 'ttr', 'hapax', 'stopword_ratio',
                          'burstiness', 'punctuation_density', 'hedging_score', 'self_reference_rate']
        
    forum_cols = [c for c in df_pd.columns if c.startswith('forum_')]
    extra_features.extend(forum_cols)
        
    present_features = [f for f in extra_features if f in df_pd.columns]
    
    if present_features:
        X_full = np.hstack([
            df_pd[present_features].values,
            embeddings
        ])
    else:
        X_full = embeddings
        
    features_filename = f"{output_dir}/{dataset_prefix}_features_full2.pkl"
    with open(features_filename, 'wb') as f:
        pickle.dump(X_full, f)
    print(f"Saved full un-split features shape: {X_full.shape} to {features_filename}")
    return df_pd
