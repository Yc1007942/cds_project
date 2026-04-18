import joblib
import os
import numpy as np
import pandas as pd
import re
import warnings
from pathlib import Path
import sys
import torch
import nltk
from datetime import datetime
from typing import List, Optional, Union

# Add project root to sys.path to allow importing from src
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import requested functions from src
try:
    from src.data_prep import (
        get_early_comments,
        engineer_comment_existence,
        engineer_early_sentiment,
        merge_engineered_features,
        save_final_features,
        extract_hour_feature,
        batch_process_nested_posts
    )
    from src.feature_engineering import (
        get_vader_compound, 
        sentiment_features, 
        engineer_text_features,
        vocab_features,
        stopword_ratio,
        punctuation_density,
        burstiness,
        hedging_score,
        self_reference_rate
    )
    from src.embeddings import (
        generate_and_save_embeddings, 
        prepare_embedding_text,
        get_bert_model_and_tokenizer,
        get_bert_embedding
    )
    print("✅ Successfully imported functions from src")
except ImportError as e:
    print(f"⚠️ Error importing from src: {e}")

# Ensure NLTK data is available
def _setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

_setup_nltk()

warnings.filterwarnings('ignore')

# Paths relative to backend/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # moltnet/
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.joblib"
BERT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "bert_classification_model_complete-20260410T155433Z-3-001" / "bert_classification_model_complete"
COMBINED_CSV_PATH = PROJECT_ROOT / "data" / "moltbook_reddit_combined10_4.csv"
# Legacy paths (kept for reference)
FEATURE_MATRIX_PATH = COMBINED_CSV_PATH
FEATURES_PATH = COMBINED_CSV_PATH

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
        self.tokenizer = None
        self.bert_model = None
        self.device = None
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

    def _init_bert(self):
        """Initialize BERT model and tokenizer if not already done"""
        if self.tokenizer is None or self.bert_model is None:
            print("📦 Initializing BERT (first time may take a while)...")
            self.tokenizer, self.bert_model, self.device = get_bert_model_and_tokenizer()
    
    def _extract_text_features(self, post_text: str, forum: str = "technology", comments: List[Union[str, dict]] = None) -> np.ndarray:
        """
        Extract features from post text for model prediction.
        Matches the 783 features expected by the model.
        """
        # 1. Comment Features (existence, avg_sentiment, max, min)
        comment_existence = 0.0
        avg_early_sentiment = 0.0
        max_early_sentiment = 0.0
        min_early_sentiment = 0.0
        
        if comments and len(comments) > 0:
            comment_existence = min(len(comments) / 10.0, 1.0)
            vader_scores = []
            for c in comments:
                content = c if isinstance(c, str) else c.get('response', '')
                vader_scores.append(get_vader_compound(content))
            
            if vader_scores:
                avg_early_sentiment = np.mean(vader_scores)
                max_early_sentiment = np.max(vader_scores)
                min_early_sentiment = np.min(vader_scores)
        
        # 2. Time Feature
        current_hour = datetime.now().hour
        
        # 3. Text Statistics
        ttr, hapax = vocab_features(post_text)
        sw_ratio = stopword_ratio(post_text)
        burst = burstiness(post_text)
        punct_density = punctuation_density(post_text)
        hedge_score = hedging_score(post_text)
        self_ref_rate = self_reference_rate(post_text)
        
        # 4. Forum One-Hot
        forum_philosophy = 1.0 if forum.lower() == "philosophy" else 0.0
        forum_technology = 1.0 if forum.lower() == "technology" else 0.0
        forum_todayilearned = 1.0 if forum.lower() == "todayilearned" else 0.0
        
        # 5. BERT Embeddings
        self._init_bert()
        embedding = get_bert_embedding(post_text, self.tokenizer, self.bert_model, self.device)
        
        feature_vector = [
            comment_existence,
            avg_early_sentiment,
            max_early_sentiment,
            min_early_sentiment,
            float(current_hour),
            ttr,
            hapax,
            sw_ratio,
            burst,
            punct_density,
            hedge_score,
            self_ref_rate,
            forum_philosophy,
            forum_technology,
            forum_todayilearned
        ]
        feature_vector.extend(embedding)
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_score_from_text(self, post_text: str, forum: str = "technology", comments: List[Union[str, dict]] = None) -> float:
        """
        Predict the engagement score directly from a post text (0-100)
        """
        if self.model is None or not post_text:
            return 50.0  # Safe fallback score
        
        try:
            X = self._extract_text_features(post_text, forum, comments)
            score = self.model.predict(X)[0]
            print(f"🔮 Predicted score for '{post_text[:20]}...': {score:.2f}")
            return float(np.clip(score, 0, 100))
        except Exception as e:
            print(f"⚠️ Prediction error: {e}. Falling back to placeholder.")
            import traceback
            traceback.print_exc()
            return 50.0

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
        if score <= 1.1:
            return max(0, min(int(round(score * 10)), 10))
        return max(0, min(int(score // 10), 10))


class BertClassificationModel:
    """BERT fine-tuned classifier for AI vs Human detection"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = BERT_CLASSIFIER_PATH
        self.tokenizer_source = None
        self.ai_label_index = 0
        self.human_label_index = 1
        self.label_names = {}
        self.feature_names = []  # kept for API compat (model-info endpoint)
        self._load()

    def _normalize_text(self, text: str) -> str:
        """Keep classifier-time normalization minimal and deterministic."""
        text = re.sub(r'http\S+|www\.\S+', ' ', text or '')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _resolve_label_mapping(self):
        """Infer class indices from saved metadata when available."""
        # The shipped classifier artifact does not include semantic id2label metadata.
        # Evaluation against the local labeled dataset shows index 0 corresponds to AI.
        self.ai_label_index = 0
        self.human_label_index = 1
        self.label_names = {}

        if self.model is not None and hasattr(self.model.config, "id2label"):
            raw_map = self.model.config.id2label or {}
            self.label_names = {int(k): str(v) for k, v in raw_map.items()}

            normalized = {idx: label.strip().lower() for idx, label in self.label_names.items()}
            for idx, label in normalized.items():
                if label in {"ai", "artificial_intelligence", "generated", "machine"}:
                    self.ai_label_index = idx
                elif label in {"human", "person", "organic"}:
                    self.human_label_index = idx

        ai_override = os.getenv("BERT_AI_LABEL_INDEX")
        human_override = os.getenv("BERT_HUMAN_LABEL_INDEX")
        if ai_override is not None:
            self.ai_label_index = int(ai_override)
        if human_override is not None:
            self.human_label_index = int(human_override)
        elif self.human_label_index == self.ai_label_index:
            self.human_label_index = 1 - self.ai_label_index

    def _load(self):
        """Load the fine-tuned BertForSequenceClassification model"""
        if not BERT_CLASSIFIER_PATH.exists():
            print(f"⚠️ BERT classifier not found at {BERT_CLASSIFIER_PATH}")
            return

        try:
            from transformers import AutoTokenizer, BertForSequenceClassification

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer_source = str(BERT_CLASSIFIER_PATH)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(BERT_CLASSIFIER_PATH),
                    local_files_only=True,
                )
            except Exception:
                self.tokenizer_source = os.getenv("BERT_TOKENIZER_PATH", "bert-base-uncased")
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_source)

            self.model = BertForSequenceClassification.from_pretrained(
                str(BERT_CLASSIFIER_PATH),
                num_labels=2
            ).to(self.device)
            self.model.eval()
            self._resolve_label_mapping()
            print(f"✅ Loaded BERT classifier from {BERT_CLASSIFIER_PATH} on {self.device}")
            print(f"   Tokenizer source: {self.tokenizer_source}")
            print(f"   Label mapping: human={self.human_label_index}, ai={self.ai_label_index}, names={self.label_names or 'generic'}")
        except Exception as e:
            print(f"⚠️ Failed to load BERT classifier: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def predict(self, text: str) -> dict:
        """Classify text as AI or Human using BERT"""
        if self.model is None or self.tokenizer is None:
            return {"label": "unknown", "prediction": -1, "confidence": 0.0, "ai_prob": 0.5, "human_prob": 0.5}

        try:
            text = self._normalize_text(text)
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            human_prob = float(probs[self.human_label_index])
            ai_prob = float(probs[self.ai_label_index])
            prediction = 1 if ai_prob > human_prob else 0

            return {
                "label": "AI" if prediction == 1 else "HUMAN",
                "prediction": prediction,
                "confidence": max(ai_prob, human_prob),
                "ai_prob": ai_prob,
                "human_prob": human_prob,
            }
        except Exception as e:
            print(f"⚠️ BERT classification error: {e}")
            import traceback
            traceback.print_exc()
            return {"label": "unknown", "prediction": -1, "confidence": 0.0, "ai_prob": 0.5, "human_prob": 0.5}


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


def get_classifier() -> BertClassificationModel:
    global _classifier
    if _classifier is None:
        _classifier = BertClassificationModel()
    return _classifier


def get_extractor() -> FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def get_features_df() -> pd.DataFrame:
    """Load and cache the features dataset (CSV)"""
    global _features_df
    if _features_df is None:
        if COMBINED_CSV_PATH.exists():
            _features_df = pd.read_csv(str(COMBINED_CSV_PATH))
            print(f"✅ Loaded features data from CSV: {_features_df.shape}")
        else:
            print(f"⚠️ Features CSV not found at {COMBINED_CSV_PATH}")
            _features_df = pd.DataFrame()
    return _features_df


def get_feature_matrix_df() -> pd.DataFrame:
    """Load and cache the feature matrix (CSV)"""
    global _feature_matrix_df
    if _feature_matrix_df is None:
        if COMBINED_CSV_PATH.exists():
            _feature_matrix_df = pd.read_csv(str(COMBINED_CSV_PATH))
            print(f"✅ Loaded feature matrix from CSV: {_feature_matrix_df.shape}")
        else:
            print(f"⚠️ Feature matrix CSV not found at {COMBINED_CSV_PATH}")
            _feature_matrix_df = pd.DataFrame()
    return _feature_matrix_df
