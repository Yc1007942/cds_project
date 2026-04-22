import joblib
import pandas as pd
import numpy as np

try:
    model = joblib.load('ai_agent_sim/backend/models/random_forest_model.joblib')
    print(f"Features in: {model.n_features_in_}")
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature Names: {list(model.feature_names_in_)}")
    else:
        print("Feature names not available")
except Exception as e:
    print(f"Error loading model: {e}")
