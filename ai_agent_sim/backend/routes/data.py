"""
Data exploration API endpoints — replaces Streamlit Data Explorer and Feature Matrix pages.
"""
from fastapi import APIRouter, Query
from typing import Optional, List
import numpy as np
import pandas as pd

from ml_model import get_features_df, get_feature_matrix_df

router = APIRouter()

# Features available in the new combined CSV
DISPLAY_COLS = ['score', 'comment_existence', 'avg_early_sentiment', 'max_early_sentiment',
                'min_early_sentiment', 'hour', 'ttr', 'hapax', 'stopword_ratio',
                'burstiness', 'punctuation_density', 'hedging_score', 'self_reference_rate',
                'forum_philosophy', 'forum_technology', 'forum_todayilearned', 'label']

PLOT_FEATURES = ['score', 'comment_existence', 'avg_early_sentiment', 'max_early_sentiment',
                 'min_early_sentiment', 'hour', 'ttr', 'hapax', 'stopword_ratio',
                 'burstiness', 'punctuation_density', 'hedging_score', 'self_reference_rate',
                 'forum_philosophy', 'forum_technology', 'forum_todayilearned']


@router.get("/stats")
async def get_stats():
    """Dataset summary metrics"""
    df = get_features_df()
    if df.empty:
        return {"total": 0, "ai_count": 0, "human_count": 0, "columns": [], "subreddits": []}

    return {
        "total": len(df),
        "ai_count": int((df['label'] == 1).sum()),
        "human_count": int((df['label'] == 0).sum()),
        "columns": [c for c in df.columns],
        "subreddits": [],  # No subreddit column in new CSV
    }


@router.get("/features")
async def get_feature_list():
    """Return list of available plottable features"""
    df = get_features_df()
    available = [f for f in PLOT_FEATURES if f in df.columns]
    return {"features": available}


@router.get("/explore")
async def get_explore_data(
    score_min: float = 0,
    score_max: float = 999999,
    label: Optional[int] = None,
    limit: int = 300,
    offset: int = 0,
):
    """Filtered and paginated data for Data Explorer"""
    df = get_features_df()
    if df.empty:
        return {"rows": [], "total_filtered": 0}

    mask = (df['score'] >= score_min) & (df['score'] <= score_max)

    if label is not None:
        mask &= (df['label'] == label)

    filtered = df[mask]
    total = len(filtered)

    cols = [c for c in DISPLAY_COLS if c in filtered.columns]
    page = filtered[cols].iloc[offset:offset + limit]

    rows = page.fillna(0).to_dict(orient='records')
    return {"rows": rows, "total_filtered": total}


@router.get("/scatter")
async def get_scatter_data(
    x: str = "score",
    y: str = "burstiness",
    sample: int = 1200,
):
    """Sampled scatter plot data"""
    df = get_features_df()
    if df.empty or x not in df.columns or y not in df.columns:
        return {"points": []}

    sampled = df.sample(min(sample, len(df)), random_state=42)
    cols = [x, y, 'label']
    cols = [c for c in cols if c in sampled.columns]

    points = sampled[cols].fillna(0).to_dict(orient='records')
    return {"points": points, "x_col": x, "y_col": y}


@router.get("/distribution")
async def get_distribution(
    feature: str = "word_count",
    bins: int = 50,
):
    """Distribution data for a feature, split by label"""
    df = get_features_df()
    if df.empty or feature not in df.columns:
        return {"bins": [], "feature": feature}

    col = df[feature].dropna()
    human = df[df['label'] == 0][feature].dropna()
    ai = df[df['label'] == 1][feature].dropna()

    # Compute histogram bins
    bin_edges = np.linspace(col.min(), col.max(), bins + 1)
    human_hist, _ = np.histogram(human, bins=bin_edges)
    ai_hist, _ = np.histogram(ai, bins=bin_edges)
    centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

    # Box plot stats
    def box_stats(s):
        if len(s) == 0:
            return {}
        return {
            "min": float(s.min()), "q1": float(s.quantile(0.25)),
            "median": float(s.median()), "q3": float(s.quantile(0.75)),
            "max": float(s.max()), "mean": float(s.mean()),
        }

    return {
        "feature": feature,
        "bins": [{"center": c, "human": int(h), "ai": int(a)}
                 for c, h, a in zip(centers, human_hist.tolist(), ai_hist.tolist())],
        "box_human": box_stats(human),
        "box_ai": box_stats(ai),
    }


@router.get("/correlation")
async def get_correlation(
    features: str = "word_count,char_count,perplexity,burstiness,sentiment_compound",
):
    """Correlation matrix for selected features"""
    df = get_features_df()
    feature_list = [f.strip() for f in features.split(',') if f.strip() in df.columns]

    if len(feature_list) < 2:
        return {"matrix": [], "labels": []}

    corr = df[feature_list].corr(numeric_only=True)
    matrix = corr.values.tolist()
    return {"matrix": matrix, "labels": feature_list}


@router.get("/clusters")
async def get_clusters(
    sample_size: int = 1200,
    n_clusters: int = 5,
    dimensions: int = 2,
):
    """PCA + KMeans clustering on AI samples"""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = get_features_df()
    if df.empty or 'label' not in df.columns:
        return {"points": [], "explained_variance": 0}

    ai_df = df[df['label'] == 1].copy()
    if len(ai_df) < 3:
        return {"points": [], "explained_variance": 0}

    # Use embedding columns if available, else numeric features
    emb_cols = [c for c in ai_df.columns if c.startswith('emb_')]
    numeric_cols = emb_cols if len(emb_cols) >= 2 else [
        c for c in ai_df.select_dtypes(include=[np.number]).columns if c != 'label'
    ]

    if len(numeric_cols) < 2:
        return {"points": [], "explained_variance": 0}

    ai_sampled = ai_df.sample(min(sample_size, len(ai_df)), random_state=42)
    X = ai_sampled[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(dimensions, 3)
    pca = PCA(n_components=n_comp, random_state=42)
    points = pca.fit_transform(X_scaled)

    k = max(2, min(n_clusters, len(ai_sampled) - 1))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(points)

    result_points = []
    for i in range(len(points)):
        p = {"cluster": int(cluster_ids[i]), "pc1": float(points[i, 0]), "pc2": float(points[i, 1])}
        if n_comp >= 3:
            p["pc3"] = float(points[i, 2])
        if 'author' in ai_sampled.columns:
            p["author"] = str(ai_sampled.iloc[i]['author'])
        if 'subreddit' in ai_sampled.columns:
            p["subreddit"] = str(ai_sampled.iloc[i]['subreddit'])
        result_points.append(p)

    return {
        "points": result_points,
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "n_clusters": k,
        "dimensions": n_comp,
    }


@router.get("/feature-stats")
async def get_feature_stats():
    """Summary statistics for features — used by radar chart in inference"""
    df = get_features_df()
    if df.empty:
        return {"human": {}, "ai": {}}

    radar_features = ['score', 'burstiness', 'ttr', 'hedging_score', 'self_reference_rate']
    available = [f for f in radar_features if f in df.columns]

    def group_means(label_val):
        subset = df[df['label'] == label_val]
        if subset.empty:
            return {f: 0.0 for f in available}
        return {f: float(subset[f].mean()) for f in available}

    def col_range(f):
        return {"min": float(df[f].min()), "max": float(df[f].max())}

    return {
        "human_means": group_means(0),
        "ai_means": group_means(1),
        "ranges": {f: col_range(f) for f in available},
        "features": available,
    }
