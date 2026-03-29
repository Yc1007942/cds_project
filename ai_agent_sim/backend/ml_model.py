import joblib
import os
import pickle
import numpy as np
from pathlib import Path

class RegressionModel:
    """Wrapper for the post score prediction regression model"""
    
    def __init__(self, model_path: str = "models/post_score_model.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load model from disk or create a placeholder if it doesn't exist"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"✅ Loaded regression model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}. Using placeholder.")
                self._create_placeholder_model()
        else:
            print(f"⚠️ Model not found at {self.model_path}. Using placeholder.")
            self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """Create a simple placeholder model for testing"""
        # This is a simple mock that returns a score based on text length
        # Replace with your actual trained model
        self.model = {
            "type": "placeholder",
            "version": "1.0"
        }
    
    def predict_score(self, post_text: str) -> float:
        """
        Predict the engagement score for a post (0-100)
        
        Args:
            post_text: The user's post text
            
        Returns:
            float: Predicted score between 0 and 100
        """
        if self.model is None:
            return self._placeholder_predict(post_text)
        
        if isinstance(self.model, dict) and self.model.get("type") == "placeholder":
            return self._placeholder_predict(post_text)
        
        try:
            # TODO: Replace with actual model prediction
            # features = self._extract_features(post_text)
            # score = self.model.predict([features])[0]
            # return float(np.clip(score, 0, 100))
            return self._placeholder_predict(post_text)
        except Exception as e:
            print(f"⚠️ Prediction error: {e}. Using placeholder.")
            return self._placeholder_predict(post_text)
    
    def _placeholder_predict(self, post_text: str) -> float:
        """
        Placeholder prediction logic based on post characteristics
        Replace this with your actual model once integrated
        """
        # Simple heuristic: longer posts tend to be more engaging
        text_length = len(post_text)
        question_bonus = 10 if "?" in post_text else 0
        exclamation_bonus = 5 if "!" in post_text else 0
        
        # Base score from text length (normalized)
        base_score = min(text_length / 3, 50)  # Max 50 from length
        
        # Add bonuses
        score = base_score + question_bonus + exclamation_bonus
        
        # Add some randomness for variety
        import random
        score += random.uniform(-5, 10)
        
        # Ensure score is in valid range
        return float(np.clip(score, 0, 100))
    
    def _extract_features(self, post_text: str) -> list:
        """
        Extract features from post text for model prediction
        TODO: Implement based on your model's requirements
        """
        features = [
            len(post_text),
            len(post_text.split()),
            post_text.count("?"),
            post_text.count("!"),
        ]
        return features
    
    def calculate_agents_to_spawn(self, score: float) -> int:
        """
        Calculate how many agents should respond based on score
        
        Score ranges:
        - 0-9: 0 agents
        - 10-19: 1 agent
        - 20-29: 2 agents
        - etc.
        """
        return max(0, int(score // 10))

# Global model instance
_model_instance = None

def get_model() -> RegressionModel:
    """Get or create the global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = RegressionModel()
    return _model_instance
