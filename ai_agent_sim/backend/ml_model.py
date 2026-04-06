import joblib
import os
import sys
import numpy as np
import pandas as pd
import torch
import nltk
from datetime import datetime
from pathlib import Path
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
    # Fallback placeholders or handle error accordingly

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

class RegressionModel:
    """Wrapper for the post score prediction regression model"""
    
    def __init__(self, model_path: str = "models/random_forest_model.joblib"):
        # Path needs to be relative to the running script (backend/)
        self.base_path = Path(__file__).parent
        self.model_path = self.base_path / model_path
        self.model = None
        self.tokenizer = None
        self.bert_model = None
        self.device = None
        
        self._load_model()
        # BERT is lazy-loaded when needed to save memory on startup
    
    def _load_model(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"✅ Loaded RandomForest model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}. Model is None.")
        else:
            print(f"⚠️ Model not found at {self.model_path}. Please check the models folder.")

    def _init_bert(self):
        """Initialize BERT model and tokenizer if not already done"""
        if self.tokenizer is None or self.bert_model is None:
            print("📦 Initializing BERT (first time may take a while)...")
            self.tokenizer, self.bert_model, self.device = get_bert_model_and_tokenizer()
    
    def _extract_features(self, post_text: str, forum: str = "technology", comments: List[Union[str, dict]] = None) -> np.ndarray:
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
        
        # 4. Forum One-Hot (philosophy, technology, todayilearned)
        forum_philosophy = 1.0 if forum.lower() == "philosophy" else 0.0
        forum_technology = 1.0 if forum.lower() == "technology" else 0.0
        forum_todayilearned = 1.0 if forum.lower() == "todayilearned" else 0.0
        
        # 5. BERT Embeddings (768)
        self._init_bert()
        # Using prepare_embedding_text approach or direct get_bert_embedding
        embedding = get_bert_embedding(post_text, self.tokenizer, self.bert_model, self.device)
        
        # Combine all features into 1D array
        # Order MUST match: ['comment_existence', 'avg_early_sentiment', 'max_early_sentiment', 'min_early_sentiment', 'hour', 'ttr', 'hapax', 'stopword_ratio', 'burstiness', 'punctuation_density', 'hedging_score', 'self_reference_rate', 'forum_philosophy', 'forum_technology', 'forum_todayilearned', 'emb_0'...'emb_767']
        
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
        
        # Add embedding components
        feature_vector.extend(embedding)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_score(self, post_text: str, forum: str = "technology", comments: List[Union[str, dict]] = None) -> float:
        """
        Predict the engagement score for a post (0-100)
        """
        if self.model is None or not post_text:
            return self._placeholder_predict(post_text)
        
        try:
            # Extract features (783 vector)
            X = self._extract_features(post_text, forum, comments)
            
            # Predict
            score = self.model.predict(X)[0]
            
            print(f"🔮 Predicted score for '{post_text[:20]}...': {score:.2f}")
            return float(np.clip(score, 0, 100))
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}. Falling back to placeholder.")
            import traceback
            traceback.print_exc()
            return self._placeholder_predict(post_text)
    
    def _placeholder_predict(self, post_text: str) -> float:
        """Fallback prediction logic"""
        if not post_text: return 0.0
        text_length = len(post_text)
        score = min(text_length / 2, 60) + 10 if "?" in post_text else 0
        import random
        score += random.uniform(0, 10)
        return float(np.clip(score, 0, 100))
    
    def calculate_agents_to_spawn(self, score: float) -> int:
        """
        Calculate how many agents should respond based on score
        """
        # More sophisticated mapping:
        # 0-20: 1-2 agents
        # 21-50: 3-5 agents
        # 51-80: 6-8 agents
        # 80+: 10 agents
        if score < 10: return 0
        if score < 30: return 2
        if score < 50: return 4
        if score < 70: return 7
        return 10

# Global model instance
_model_instance = None

def get_model() -> RegressionModel:
    """Get or create the global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = RegressionModel()
    return _model_instance
