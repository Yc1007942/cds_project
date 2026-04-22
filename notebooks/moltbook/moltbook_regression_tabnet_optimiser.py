import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from pytorch_tabnet import TabNetRegressor

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "../../data/moltbook_with_keyword_features.pkl"
EMBEDDING_COL = "embeddings"
TARGET_COL = "score"
RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE_FROM_TEMP = 0.50
DROP_COLS = ["safe_content", "content", "id"]

# PCA configuration
PCA_COMPONENTS = 128  # Adjust based on your embedding dimension

# TabNet hyperparameters
TABNET_PARAMS = {
    'n_d': 64,
    'n_a': 64,
    'n_steps': 8,
    'gamma': 1.5,
    'n_independent': 2,
    'n_shared': 2,
    'momentum': 0.02,
    'mask_type': 'sparsemax',
    'lambda_sparse': 5e-3,
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
    'patience': 30,
    'loss_fn': torch.nn.MSELoss(),
    'eval_metric': ['rmse'],
}

# ==========================================
# 1. Load and prepare data with time features
# ==========================================
print("Loading data...")
moltbook = pd.read_pickle(DATA_PATH)
print(f"Original shape: {moltbook.shape}")

# ==========================================
# 1a. Feature Engineering: Time of Day (Option C - Both methods)
# ==========================================
print("\nEngineering time features...")

# Method 1: Sine/Cosine encoding (cyclical)
hour_rad = 2 * np.pi * moltbook['hour'] / 24
moltbook['hour_sin'] = np.sin(hour_rad)
moltbook['hour_cos'] = np.cos(hour_rad)

# Method 2: Categorical time-of-day bins
def time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'night'

moltbook['tod'] = moltbook['hour'].apply(time_of_day)
tod_dummies = pd.get_dummies(moltbook['tod'], prefix='tod')

# Expand embeddings
print("\nProcessing embeddings...")
embedding_lists = moltbook[EMBEDDING_COL].values
lengths = [len(lst) for lst in embedding_lists]
if len(set(lengths)) != 1:
    raise ValueError("Embedding lists have varying lengths.")
emb_dim = lengths[0]
print(f"Original embedding dimension: {emb_dim}")

emb_df = pd.DataFrame(
    np.vstack(embedding_lists),
    index=moltbook.index,
    columns=[f"emb_{i}" for i in range(emb_dim)]
)

# Base features
KEPT_FEATURES = [
    "comment_existence",
    "max_early_sentiment", 
    "avg_early_sentiment",
    "min_early_sentiment",
    "punctuation_density",
    "ttr",
    "has_biological_tax",
    "has_lobster",
    "has_great_lobster"
]

# Add engineered time features
available_features = [f for f in KEPT_FEATURES if f in moltbook.columns]
X_base = moltbook[available_features].copy()

# Add sine/cosine features (drop original hour)
X_base['hour_sin'] = moltbook['hour_sin']
X_base['hour_cos'] = moltbook['hour_cos']

# Add time-of-day dummies
X_base = pd.concat([X_base, tod_dummies], axis=1)

# Target (log transform)
y_raw = moltbook[TARGET_COL].clip(lower=0)
y = np.log1p(y_raw)

print(f"\nFinal feature count (excluding embeddings): {X_base.shape[1]}")
print(f"Features: {list(X_base.columns)}")

# ==========================================
# 2. Train/Validation/Test Split (same split for both models)
# ==========================================
print("\nSplitting data...")
X_base_train, X_base_temp, emb_train, emb_temp, y_train, y_temp = train_test_split(
    X_base, emb_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_base_val, X_base_test, emb_val, emb_test, y_val, y_test = train_test_split(
    X_base_temp, emb_temp, y_temp, test_size=VAL_SIZE_FROM_TEMP, random_state=RANDOM_STATE
)

print(f"Training set size: {len(X_base_train)}")
print(f"Validation set size: {len(X_base_val)}")
print(f"Test set size: {len(X_base_test)}")

# ==========================================
# Helper function to train and evaluate model
# ==========================================
def train_and_evaluate(emb_train_df, emb_val_df, emb_test_df, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print('='*60)
    
    # Combine base features with embeddings
    X_train = pd.concat([X_base_train, emb_train_df], axis=1)
    X_val = pd.concat([X_base_val, emb_val_df], axis=1)
    X_test = pd.concat([X_base_test, emb_test_df], axis=1)
    
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Convert to numpy float32
    X_train_np = X_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
    y_val_np = y_val.values.reshape(-1, 1).astype(np.float32)
    y_test_np = y_test.values.reshape(-1, 1).astype(np.float32)
    
    # Apply RobustScaler
    print("\nApplying RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)
    X_test_scaled = scaler.transform(X_test_np)
    
    # Train TabNet Model
    print("\nInitializing TabNet Regressor...")
    model = TabNetRegressor(**TABNET_PARAMS)
    
    print("Starting training...")
    model.fit(
        X_train=X_train_scaled,
        y_train=y_train_np,
        eval_set=[(X_val_scaled, y_val_np)],
        eval_name=['valid'],
        **TRAIN_PARAMS
    )
    
    # Evaluation
    y_val_pred = model.predict(X_val_scaled).flatten()
    y_test_pred = model.predict(X_test_scaled).flatten()
    
    r2_val = r2_score(y_val_np.flatten(), y_val_pred)
    r2_test = r2_score(y_test_np.flatten(), y_test_pred)
    
    # Original scale evaluation
    y_val_original = np.expm1(y_val_np.flatten())
    y_val_pred_original = np.expm1(y_val_pred)
    y_test_original = np.expm1(y_test_np.flatten())
    y_test_pred_original = np.expm1(y_test_pred)
    
    r2_val_original = r2_score(y_val_original, y_val_pred_original)
    r2_test_original = r2_score(y_test_original, y_test_pred_original)
    
    # Results
    print(f"\n{'='*60}")
    print(f"{model_name} - RESULTS")
    print('='*60)
    print(f"Log-transformed scale:")
    print(f"  Validation R²: {r2_val:.4f}")
    print(f"  Test R²: {r2_test:.4f}")
    print(f"\nOriginal scale:")
    print(f"  Validation R²: {r2_val_original:.4f}")
    print(f"  Test R²: {r2_test_original:.4f}")
    print('='*60)
    
    return {
        'name': model_name,
        'r2_val': r2_val,
        'r2_test': r2_test,
        'r2_val_original': r2_val_original,
        'r2_test_original': r2_test_original,
        'model': model,
        'scaler': scaler,
        'feature_dim': X_train.shape[1]
    }

# ==========================================
# 3. Run Model WITHOUT PCA
# ==========================================
results = {}

# Use original embeddings
results['no_pca'] = train_and_evaluate(
    emb_train, emb_val, emb_test, 
    model_name="Model WITHOUT PCA"
)

# ==========================================
# 4. Apply PCA and Run Model WITH PCA
# ==========================================
if emb_dim > PCA_COMPONENTS:
    print(f"\n{'='*60}")
    print(f"Applying PCA to reduce embeddings from {emb_dim} to {PCA_COMPONENTS} dimensions...")
    print('='*60)
    
    # Fit PCA on training embeddings only
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    emb_train_pca = pca.fit_transform(emb_train)
    emb_val_pca = pca.transform(emb_val)
    emb_test_pca = pca.transform(emb_test)
    
    # Convert to DataFrame
    emb_train_pca_df = pd.DataFrame(
        emb_train_pca,
        index=emb_train.index,
        columns=[f"emb_pca_{i}" for i in range(PCA_COMPONENTS)]
    )
    emb_val_pca_df = pd.DataFrame(
        emb_val_pca,
        index=emb_val.index,
        columns=[f"emb_pca_{i}" for i in range(PCA_COMPONENTS)]
    )
    emb_test_pca_df = pd.DataFrame(
        emb_test_pca,
        index=emb_test.index,
        columns=[f"emb_pca_{i}" for i in range(PCA_COMPONENTS)]
    )
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"PCA components: {PCA_COMPONENTS}")
    
    # Train model with PCA embeddings
    results['with_pca'] = train_and_evaluate(
        emb_train_pca_df, emb_val_pca_df, emb_test_pca_df, 
        model_name="Model WITH PCA"
    )
    
    # Save PCA object
    joblib.dump(pca, 'embedding_pca.pkl')
    print("\nPCA object saved as 'embedding_pca.pkl'")
    
else:
    print(f"\nWarning: Embedding dimension ({emb_dim}) <= PCA components ({PCA_COMPONENTS}). Skipping PCA.")
    results['with_pca'] = None

# ==========================================
# 5. Save Models and Preprocessing Objects
# ==========================================
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

# # Save models
# joblib.dump(results['no_pca']['model'], 'tabnet_model_no_pca.pkl')
# joblib.dump(results['no_pca']['scaler'], 'robust_scaler_no_pca.pkl')
# print("✓ Saved: tabnet_model_no_pca.pkl")
# print("✓ Saved: robust_scaler_no_pca.pkl")

# if results['with_pca'] is not None:
#     joblib.dump(results['with_pca']['model'], 'tabnet_model_with_pca.pkl')
#     joblib.dump(results['with_pca']['scaler'], 'robust_scaler_with_pca.pkl')
#     print("✓ Saved: tabnet_model_with_pca.pkl")
#     print("✓ Saved: robust_scaler_with_pca.pkl")

# ==========================================
# 6. Final Comparison Summary
# ==========================================
print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)

print("\nFeature Dimensions:")
print(f"  Without PCA: {results['no_pca']['feature_dim']} features")
if results['with_pca'] is not None:
    print(f"  With PCA:    {results['with_pca']['feature_dim']} features")
    print(f"  Reduction:   {results['no_pca']['feature_dim'] - results['with_pca']['feature_dim']} features ({(1 - results['with_pca']['feature_dim']/results['no_pca']['feature_dim'])*100:.1f}%)")

print("\nTest R² Comparison (Original Scale):")
print(f"  Without PCA: {results['no_pca']['r2_test_original']:.4f}")
if results['with_pca'] is not None:
    print(f"  With PCA:    {results['with_pca']['r2_test_original']:.4f}")
    improvement = results['with_pca']['r2_test_original'] - results['no_pca']['r2_test_original']
    print(f"  Difference:  {improvement:+.4f}")

print("\nTest R² Comparison (Log Scale):")
print(f"  Without PCA: {results['no_pca']['r2_test']:.4f}")
if results['with_pca'] is not None:
    print(f"  With PCA:    {results['with_pca']['r2_test']:.4f}")
    improvement = results['with_pca']['r2_test'] - results['no_pca']['r2_test']
    print(f"  Difference:  {improvement:+.4f}")

# ==========================================
# 7. Visualization Comparison
# ==========================================
fig, axes = plt.subplots(1, 2 if results['with_pca'] is not None else 1, figsize=(12, 5))

# Plot without PCA
axes[0].scatter(np.expm1(y_test), np.expm1(results['no_pca']['model'].predict(
    results['no_pca']['scaler'].transform(
        pd.concat([X_base_test, emb_test], axis=1).values.astype(np.float32)
    )
).flatten()), alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0].plot([np.expm1(y_test).min(), np.expm1(y_test).max()], 
             [np.expm1(y_test).min(), np.expm1(y_test).max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Score')
axes[0].set_ylabel('Predicted Score')
axes[0].set_title(f'Without PCA\nTest R² = {results["no_pca"]["r2_test_original"]:.4f}')
axes[0].grid(True, alpha=0.3)

# Plot with PCA
if results['with_pca'] is not None:
    emb_test_pca_np = results['with_pca']['model'].scaler.transform(
        pd.concat([X_base_test, emb_test_pca_df], axis=1).values.astype(np.float32)
    )
    axes[1].scatter(np.expm1(y_test), np.expm1(results['with_pca']['model'].predict(emb_test_pca_np).flatten()), 
                   alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1].plot([np.expm1(y_test).min(), np.expm1(y_test).max()], 
                 [np.expm1(y_test).min(), np.expm1(y_test).max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Score')
    axes[1].set_ylabel('Predicted Score')
    axes[1].set_title(f'With PCA\nTest R² = {results["with_pca"]["r2_test_original"]:.4f}')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("COMPARISON COMPLETE")
print("="*60)
print("\nFiles saved:")
print("  - tabnet_model_no_pca.pkl")
print("  - robust_scaler_no_pca.pkl")
if results['with_pca'] is not None:
    print("  - tabnet_model_with_pca.pkl")
    print("  - robust_scaler_with_pca.pkl")
    print("  - embedding_pca.pkl")
print("  - model_comparison.png")
print("\n" + "="*60)