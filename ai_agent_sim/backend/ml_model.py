import joblib
import os
import numpy as np
import pandas as pd
import re
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Paths relative to backend/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # moltnet/
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.joblib"
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "feature_matrix_full_subset_train.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "features_subset_train.parquet"

# Columns to exclude from training features
NON_FEATURE_COLS = {
    'label', 'id', 'split', 'author', 'subreddit', 'text',
    'created_utc', 'interaction_type', 'source', 'is_comment',
    'text_clean', 'timestamp', 'hour_of_day', 'day_of_week',
    'post_id', 'upvotes'
}


class EngagementRegressor:
    """Wrapper for the pre-trained RandomForestRegressor (engagement scoring)"""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            try:
                self.model = joblib.load(str(MODEL_PATH))
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = list(self.model.feature_names_in_)
                print(f"✅ Loaded engagement regressor from {MODEL_PATH}")
                print(f"   Features: {self.model.n_features_in_}, Estimators: {self.model.n_estimators}")
            except Exception as e:
                print(f"⚠️ Failed to load regressor: {e}")
                self.model = None
        else:
            print(f"⚠️ Regressor model not found at {MODEL_PATH}")

    def predict_score(self, features_df: pd.DataFrame) -> float:
        """Predict engagement score from a feature DataFrame"""
        if self.model is None:
            return self._placeholder_predict(features_df)

        try:
            # Align columns to model's expected features
            X = pd.DataFrame(columns=self.feature_names)
            for col in self.feature_names:
                X[col] = features_df[col].values if col in features_df.columns else 0.0
            X = X.fillna(0).astype(float)
            score = float(self.model.predict(X)[0])
            return float(np.clip(score, 0, 100))
        except Exception as e:
            print(f"⚠️ Regressor prediction error: {e}")
            return self._placeholder_predict(features_df)

    def _placeholder_predict(self, features_df: pd.DataFrame) -> float:
        """Fallback heuristic"""
        import random
        word_count = features_df.get('word_count', pd.Series([10])).iloc[0]
        base = min(float(word_count) / 3, 50)
        return float(np.clip(base + random.uniform(-5, 15), 0, 100))

    def calculate_agents_to_spawn(self, score: float) -> int:
        # The regressor appears to output a value between 0.0 and 1.0.
        # If the score is <= 1.0, scale it up to 10.
        # If the score is already > 1.0 (e.g. out of 100), divide by 10.
        if score <= 1.0:
            return max(0, min(int(round(score * 10)), 10))
        return max(0, min(int(score // 10), 10))


class ClassificationModel:
    """RandomForestClassifier for AI vs Human detection"""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self._train()

    def _train(self):
        """Train classifier from feature matrix parquet"""
        if not FEATURE_MATRIX_PATH.exists():
            print(f"⚠️ Feature matrix not found at {FEATURE_MATRIX_PATH}")
            return

        try:
            from sklearn.ensemble import RandomForestClassifier

            df = pd.read_parquet(str(FEATURE_MATRIX_PATH))
            features = [c for c in df.columns if c not in NON_FEATURE_COLS]
            X = df[features].select_dtypes(include=[np.number]).fillna(0)
            y = df['label']

            self.feature_names = X.columns.tolist()
            self.model = RandomForestClassifier(
                n_estimators=50, random_state=42, max_depth=10, n_jobs=-1
            )
            self.model.fit(X, y)
            print(f"✅ Trained classifier: {len(self.feature_names)} features, {len(X)} samples")
        except Exception as e:
            print(f"⚠️ Failed to train classifier: {e}")
            self.model = None

    def predict(self, features_df: pd.DataFrame) -> dict:
        """Classify text as AI or Human"""
        if self.model is None:
            return {"label": "unknown", "confidence": 0.0, "ai_prob": 0.5, "human_prob": 0.5}

        try:
            X = pd.DataFrame(columns=self.feature_names)
            for col in self.feature_names:
                X[col] = features_df[col].values if col in features_df.columns else 0.0
            X = X.fillna(0).astype(float)

            prediction = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            ai_prob = float(proba[1]) if len(proba) > 1 else 0.0
            human_prob = float(proba[0])

            return {
                "label": "AI" if prediction == 1 else "HUMAN",
                "prediction": prediction,
                "confidence": max(ai_prob, human_prob),
                "ai_prob": ai_prob,
                "human_prob": human_prob,
            }
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return {"label": "unknown", "confidence": 0.0, "ai_prob": 0.5, "human_prob": 0.5}


class FeatureExtractor:
    """Extract features from raw text for model inference"""

    def __init__(self):
        self.st_model = None
        self._load_sentence_transformer()

    def _load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Loaded sentence transformer (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"⚠️ Failed to load sentence transformer: {e}")

    def extract(self, text: str) -> pd.DataFrame:
        """Extract full feature vector from raw text"""
        text_clean = re.sub(r'http\S+|www\.\S+', '', text)
        text_clean = re.sub(r'[ \t]+', ' ', text_clean).strip()

        words = text_clean.split()
        word_count = len(words)
        char_count = len(text_clean)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text_clean) if s.strip()]
        sentence_count = max(len(sentences), 1)

        feature_dict = {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': word_count / sentence_count,
            'sentence_length_std': np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0,
            'paragraph_count': max(text_clean.count('\n\n') + 1, 1),
            'flesch_kincaid': 8.0,
            'gunning_fog': 8.0,
            'coleman_liau': 8.0,
            'automated_readability': 8.0,
            'ttr': len(set(words)) / max(word_count, 1),
            'hapax_ratio': sum(1 for w in set(words) if words.count(w) == 1) / max(len(set(words)), 1),
            'stopword_ratio': 0.4,
            'punctuation_density': sum(1 for c in text_clean if not c.isalnum() and not c.isspace()) / max(char_count, 1),
            'exclamation_count': text_clean.count('!'),
            'question_count': text_clean.count('?'),
            'ellipsis_count': text_clean.count('...'),
            'emoji_count': 0,
            'trigram_repeat_rate': 0.0,
            'fourgram_repeat_rate': 0.0,
            'hedging_per_100w': 0.0,
            'self_reference_rate': sum(1 for w in words if w.lower() in ('i', 'me', 'my', 'mine', 'myself')) / max(word_count, 1),
            'formality_score': 50.0,
            'author_post_count': 1.0,
            'author_mean_word_count': float(word_count),
            'author_std_word_count': 0.0,
            'author_mean_upvotes': 1.0,
            'author_community_diversity': 1.0,
            'author_mean_hedging': 0.0,
            'inter_post_median': 0.0,
            'inter_post_std': 0.0,
            'perplexity': 50.0,
            'burstiness': 0.5,
            'sentiment_compound': 0.0,
            'sentiment_pos': 0.0,
            'sentiment_neg': 0.0,
            'sentiment_neu': 1.0,
            'sentiment_variability': 0.0,
            'topic_id': 0,
        }

        # Try to compute readability with textstat
        try:
            import textstat
            feature_dict['flesch_kincaid'] = textstat.flesch_kincaid_grade(text_clean)
            feature_dict['gunning_fog'] = textstat.gunning_fog(text_clean)
            feature_dict['coleman_liau'] = textstat.coleman_liau_index(text_clean)
            feature_dict['automated_readability'] = textstat.automated_readability_index(text_clean)
        except Exception:
            pass

        # Try sentiment with VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text_clean)
            feature_dict['sentiment_compound'] = scores['compound']
            feature_dict['sentiment_pos'] = scores['pos']
            feature_dict['sentiment_neg'] = scores['neg']
            feature_dict['sentiment_neu'] = scores['neu']
        except Exception:
            pass

        # Add POS tag features (stub zeros for now)
        pos_tags = ['pos_NN', 'pos_NNS', 'pos_NNP', 'pos_VB', 'pos_VBD', 'pos_VBG',
                     'pos_VBN', 'pos_VBP', 'pos_VBZ', 'pos_JJ', 'pos_JJR', 'pos_JJS',
                     'pos_RB', 'pos_RBR', 'pos_RBS', 'pos_PRP', 'pos_DT', 'pos_IN', 'pos_CC']
        for tag in pos_tags:
            feature_dict[tag] = 0.0

        # Sentence transformer embeddings
        if self.st_model is not None:
            try:
                embeddings = self.st_model.encode([text_clean])[0]
                for i, emb in enumerate(embeddings):
                    feature_dict[f'emb_{i}'] = float(emb)
            except Exception:
                pass

        return pd.DataFrame([feature_dict])


# ─── Global Singletons ──────────────────────────────────────────────
_regressor = None
_classifier = None
_extractor = None
_features_df = None
_feature_matrix_df = None


def get_regressor() -> EngagementRegressor:
    global _regressor
    if _regressor is None:
        _regressor = EngagementRegressor()
    return _regressor


def get_classifier() -> ClassificationModel:
    global _classifier
    if _classifier is None:
        _classifier = ClassificationModel()
    return _classifier


def get_extractor() -> FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def get_features_df() -> pd.DataFrame:
    """Load and cache the features dataset"""
    global _features_df
    if _features_df is None:
        if FEATURES_PATH.exists():
            _features_df = pd.read_parquet(str(FEATURES_PATH))
            print(f"✅ Loaded features data: {_features_df.shape}")
        else:
            print(f"⚠️ Features data not found at {FEATURES_PATH}")
            _features_df = pd.DataFrame()
    return _features_df


def get_feature_matrix_df() -> pd.DataFrame:
    """Load and cache the feature matrix"""
    global _feature_matrix_df
    if _feature_matrix_df is None:
        if FEATURE_MATRIX_PATH.exists():
            _feature_matrix_df = pd.read_parquet(str(FEATURE_MATRIX_PATH))
            print(f"✅ Loaded feature matrix: {_feature_matrix_df.shape}")
        else:
            print(f"⚠️ Feature matrix not found at {FEATURE_MATRIX_PATH}")
            _feature_matrix_df = pd.DataFrame()
    return _feature_matrix_df
