import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from pytorch_tabnet import TabNetRegressor

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "../data/processed_v1_5_4_new_full.pkl"
EMBEDDING_COL = "embeddings"
TARGET_COL = "score"
RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE_FROM_TEMP = 0.50
DROP_COLS = ["safe_content", "content", "id"]

# TabNet hyperparameters
TABNET_PARAMS = {
    'n_d': 64,
    'n_a': 64,
    'n_steps': 5,
    'gamma': 1.5,
    'n_independent': 2,
    'n_shared': 2,
    'momentum': 0.02,
    'mask_type': 'sparsemax',
    'verbose': 10,
    'seed': RANDOM_STATE,
    'device_name': 'auto',
}

TRAIN_PARAMS = {
    'batch_size': 1024,
    'virtual_batch_size': 128,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 500,
    'patience': 50,
    'loss_fn': torch.nn.MSELoss(),        # MSE loss for training
    'eval_metric': ['rmse'],              # built-in RMSE for early stopping
}

# ==========================================
# 1. Load and prepare data
# ==========================================
print("Loading data...")
moltbook = pd.read_pickle(DATA_PATH)
print(f"Original shape: {moltbook.shape}")

# Expand embeddings
embedding_lists = moltbook[EMBEDDING_COL].values
lengths = [len(lst) for lst in embedding_lists]
if len(set(lengths)) != 1:
    raise ValueError("Embedding lists have varying lengths.")
emb_dim = lengths[0]
print(f"Embedding dimension: {emb_dim}")

emb_df = pd.DataFrame(
    np.vstack(embedding_lists),
    index=moltbook.index,
    columns=[f"emb_{i}" for i in range(emb_dim)]
)

# Base features and target
X_base = moltbook.drop(columns=[TARGET_COL, EMBEDDING_COL] + DROP_COLS)
y_raw = moltbook[TARGET_COL].clip(lower=0)
y = np.log1p(y_raw)

# Train/val/test split (same as RF)
X_base_train, X_base_temp, emb_train, emb_temp, y_train, y_temp = train_test_split(
    X_base, emb_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_base_val, X_base_test, emb_val, emb_test, y_val, y_test = train_test_split(
    X_base_temp, emb_temp, y_temp, test_size=VAL_SIZE_FROM_TEMP, random_state=RANDOM_STATE
)

X_train = pd.concat([X_base_train, emb_train], axis=1)
X_val   = pd.concat([X_base_val, emb_val], axis=1)
X_test  = pd.concat([X_base_test, emb_test], axis=1)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Convert to numpy float32
X_train_np = X_train.values.astype(np.float32)
X_val_np   = X_val.values.astype(np.float32)
X_test_np  = X_test.values.astype(np.float32)
y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
y_val_np   = y_val.values.reshape(-1, 1).astype(np.float32)
y_test_np  = y_test.values.reshape(-1, 1).astype(np.float32)

# ==========================================
# 2. Train TabNet
# ==========================================
print("\nInitializing TabNet Regressor...")
model = TabNetRegressor(**TABNET_PARAMS)

print("Starting training (early stopping on RMSE)...")
model.fit(
    X_train=X_train_np,
    y_train=y_train_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_name=['valid'],
    **TRAIN_PARAMS
)

# ==========================================
# 3. Final R² evaluation (variance explained)
# ==========================================
y_val_pred = model.predict(X_val_np).flatten()
y_test_pred = model.predict(X_test_np).flatten()

r2_val = r2_score(y_val_np.flatten(), y_val_pred)
r2_test = r2_score(y_test_np.flatten(), y_test_pred)

print(f"\nValidation R²: {r2_val:.4f}")
print(f"Test R²: {r2_test:.4f}")

# ==========================================
# 4. Feature importancefrom sklearn.inspection import permutation_importance

# After training your model
result = permutation_importance(
    model,                      # your trained TabNet model
    X_val_np,                   # validation features
    y_val_np.flatten(),         # validation targets
    n_repeats=5,                # number of times to shuffle each feature
    scoring='r2',               # use R² as the scoring metric
    random_state=RANDOM_STATE,
    n_jobs=-1                   # use all CPU cores for speed
)

# Extract importance scores
importances = result.importances_mean
std_devs = result.importances_std

# Create importance DataFrame
imp_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances,
    'StdDev': std_devs
}).sort_values('Importance', ascending=False).head(20)

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1], 
         xerr=imp_df['StdDev'][::-1], color='steelblue', capsize=3)
plt.xlabel('Permutation Importance (Drop in R²)')
plt.title('Top 20 Features - Permutation Importance')
plt.tight_layout()
plt.show()