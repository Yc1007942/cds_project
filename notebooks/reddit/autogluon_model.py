#!/usr/bin/env python3
"""
Regression with AutoGluon on Moltbook data.
Modified to match FT-Transformer's feature selection for fair comparison.
"""
import os
import datetime
import shutil
import gc


# Install AutoGluon

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from autogluon.tabular import TabularDataset, TabularPredictor

# ==========================================
# CONFIGURATION
# ==========================================
DESTINATION_DIR = '../../data'
# Update this path to match your FT-Transformer data file
DATA_PATH = f"{DESTINATION_DIR}/reddit_11_4_full.pkl"

EMBEDDING_COL = "embeddings"
TARGET_COL = "score"
RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE_FROM_TEMP = 0.50
DROP_COLS = ["safe_content", "content", "id"]

# Match FT-Transformer's feature selection
# KEPT_FEATURES = [
#     "comment_existence",
#     "max_early_sentiment", 
#     "avg_early_sentiment",
#     "min_early_sentiment",
#     "punctuation_density",
#     "ttr",
#     "hour",
#     "has_biological_tax",
#     "has_lobster",
#     "has_great_lobster"
# ]

# AutoGluon settings
PRESETS = 'best_quality'  # Can also try 'medium_quality' for faster results
TIME_LIMIT = 3600          # seconds (None for unlimited)
EVAL_METRIC = 'r2'

# ==========================================
# 🔧 PRE-RUN CHECK & CLEANUP
# ==========================================
print("\n" + "="*50)
print("CLEANING UP PREVIOUS STATE")
print("="*50)

# Delete any existing predictor object
try:
    if 'predictor' in globals():
        del predictor
        gc.collect()
        print("Deleted previous predictor object.")
except:
    pass

# Create unique paths with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
MODEL_PATH = f"/content/drive/MyDrive/autogluon_models_{timestamp}"
RESULTS_PATH = f"/content/drive/MyDrive/autogluon_results_{timestamp}"

# Ensure clean directory
if os.path.exists(MODEL_PATH):
    print(f"Deleting existing directory: {MODEL_PATH}")
    shutil.rmtree(MODEL_PATH)

print(f"Model will be saved to: {MODEL_PATH}")
print(f"Results will be saved to: {RESULTS_PATH}.pkl")

# ==========================================
# 1. Load and prepare data
# ==========================================
print("\n" + "="*50)
print("LOADING AND PREPARING DATA")
print("="*50)

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

# ==========================================
# FEATURE SELECTION (Match FT-Transformer)
# ==========================================
# Filter to available features (some might not exist in your data)
# available_features = [f for f in KEPT_FEATURES if f in moltbook.columns]
# print(f"\nKeeping {len(available_features)} non-embedding features:")
# for feature in available_features:
#     print(f"  - {feature}")

# # Check for missing features
# missing_features = set(KEPT_FEATURES) - set(available_features)
# if missing_features:
#     print(f"\n⚠️ Warning: These features not found in data: {missing_features}")

# # Select only the kept features
# X_base = moltbook[available_features].copy()




X_base = moltbook.drop(columns=[TARGET_COL, EMBEDDING_COL,'forum','created_utc_dt','title', 'selftext','safe_content',"content"])


forum_columns = ['forum_philosophy', 'forum_technology', 'forum_todayilearned']
existing_forum_cols = [col for col in forum_columns if col in X_base.columns]
if existing_forum_cols:
    X_base['forum'] = 'other'
    for col in existing_forum_cols:
        forum_name = col.replace('forum_', '')
        X_base.loc[X_base[col] == 1, 'forum'] = forum_name
    X_base = X_base.drop(columns=existing_forum_cols)

# ==========================================
# HANDLE HOUR AS CATEGORICAL (AutoGluon can handle automatically, but explicit is fine)
# ==========================================
if 'hour' in X_base.columns:
    X_base['hour'] = X_base['hour'].astype(int)
    X_base['hour_category'] = X_base['hour'].astype(str)
    X_base = X_base.drop(columns=['hour'])

# Identify categorical columns (none in our selected features except maybe forum)
categorical_cols = []
for col in X_base.columns:
    if col in ['forum', 'hour_category'] or X_base[col].dtype in ['object', 'category']:
        categorical_cols.append(col)
        X_base[col] = X_base[col].astype('category')



numerical_cols = [col for col in X_base.columns if col not in categorical_cols]

print(f"\nCategorical columns: {categorical_cols if categorical_cols else 'None'}")
print(f"Numerical columns: {len(numerical_cols)}")
# print(f"Total features: {len(available_features)} non-embedding + {emb_dim} embedding = {len(available_features) + emb_dim}")

# ==========================================
# Target transformation (Match FT-Transformer)
# ==========================================
y_raw = moltbook[TARGET_COL].clip(lower=0)
y = np.log1p(y_raw)  # log-transform target

# ==========================================
# Train / Validation / Test Split (Identical to FT-Transformer)
# ==========================================
X_base_train, X_base_temp, emb_train, emb_temp, y_train, y_temp = train_test_split(
    X_base, emb_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_base_val, X_base_test, emb_val, emb_test, y_val, y_test = train_test_split(
    X_base_temp, emb_temp, y_temp, test_size=VAL_SIZE_FROM_TEMP, random_state=RANDOM_STATE
)

# Reset indices
X_base_train = X_base_train.reset_index(drop=True)
X_base_val   = X_base_val.reset_index(drop=True)
X_base_test  = X_base_test.reset_index(drop=True)
emb_train    = emb_train.reset_index(drop=True)
emb_val      = emb_val.reset_index(drop=True)
emb_test     = emb_test.reset_index(drop=True)
y_train      = y_train.reset_index(drop=True)
y_val        = y_val.reset_index(drop=True)
y_test       = y_test.reset_index(drop=True)

# Combine into DataFrames
train_df = pd.concat([X_base_train, emb_train, y_train], axis=1)
val_df   = pd.concat([X_base_val,   emb_val,   y_val],   axis=1)
test_df  = pd.concat([X_base_test,  emb_test,  y_test],  axis=1)

# Convert to AutoGluon's TabularDataset
train_data = TabularDataset(train_df)
val_data   = TabularDataset(val_df)
test_data  = TabularDataset(test_df)

print(f"\nTraining data: {train_data.shape}")
print(f"Validation data: {val_data.shape}")
print(f"Test data: {test_data.shape}")

# ==========================================
# 3. Train AutoGluon Predictor
# ==========================================
print("\n" + "="*50)
print("TRAINING AUTOGLUON PREDICTOR")
print("="*50)

# Create fresh predictor instance
predictor = TabularPredictor(
    label=TARGET_COL,
    problem_type='regression',
    eval_metric=EVAL_METRIC,
    path=MODEL_PATH
)

# Fit the model
predictor.fit(
    train_data=train_data,
    tuning_data=val_data,
    presets=PRESETS,
    time_limit=TIME_LIMIT,
    verbosity=2,
    use_bag_holdout=True,          # Allows tuning_data with bagging
    dynamic_stacking=False,        # Disable DyStack to avoid internal sub-fit bug
    num_stack_levels=1,            # Use 1 level of stacking
    holdout_frac=None
)

# ==========================================
# 4. Evaluate on test set
# ==========================================
print("\n" + "="*50)
print("EVALUATION ON TEST SET")
print("="*50)

# Get leaderboard
leaderboard = predictor.leaderboard(test_data, silent=True)
print("\nModel Leaderboard:")
print(leaderboard)

# Predictions with best model
test_preds = predictor.predict(test_data, model=predictor.model_best)
test_true  = test_data[TARGET_COL].values

# Metrics in log space
test_r2_log   = r2_score(test_true, test_preds)
test_rmse_log = np.sqrt(mean_squared_error(test_true, test_preds))
test_mae_log  = mean_absolute_error(test_true, test_preds)

print(f"\n📊 Log Space Metrics:")
print(f"  Test R²:  {test_r2_log:.4f}")
print(f"  Test RMSE: {test_rmse_log:.4f}")
print(f"  Test MAE:  {test_mae_log:.4f}")

# Metrics in original space
test_preds_orig = np.expm1(test_preds)
test_true_orig  = np.expm1(test_true)
test_r2_orig    = r2_score(test_true_orig, test_preds_orig)
test_rmse_orig  = np.sqrt(mean_squared_error(test_true_orig, test_preds_orig))
test_mae_orig   = mean_absolute_error(test_true_orig, test_preds_orig)

print(f"\n📊 Original Space Metrics:")
print(f"  Test R²:  {test_r2_orig:.4f}")
print(f"  Test RMSE: {test_rmse_orig:.2f}")
print(f"  Test MAE:  {test_mae_orig:.2f}")

# ==========================================
# 5. Validation set evaluation
# ==========================================
val_preds = predictor.predict(val_data, model=predictor.model_best)
val_true  = val_data[TARGET_COL].values

val_r2_log = r2_score(val_true, val_preds)
val_r2_orig = r2_score(np.expm1(val_true), np.expm1(val_preds))

print(f"\n📊 Validation Metrics:")
print(f"  Validation R² (log): {val_r2_log:.4f}")
print(f"  Validation R² (original): {val_r2_orig:.4f}")

# ==========================================
# 6. Save results
# ==========================================
results = {
    'test_log': {'r2': test_r2_log, 'rmse': test_rmse_log, 'mae': test_mae_log},
    'test_orig': {'r2': test_r2_orig, 'rmse': test_rmse_orig, 'mae': test_mae_orig},
    'val_log': {'r2': val_r2_log},
    'val_orig': {'r2': val_r2_orig},
    'best_model': predictor.model_best,
    'model_path': MODEL_PATH,
    # 'features_used': available_features,
    # 'total_features': len(available_features) + emb_dim,
    'embedding_dim': emb_dim,
    'leaderboard': leaderboard.to_dict(),
    'config': {
        'presets': PRESETS,
        'time_limit': TIME_LIMIT,
        'eval_metric': EVAL_METRIC,
    }
}
joblib.dump(results, f"{RESULTS_PATH}.pkl")
print(f"\n💾 Results saved to {RESULTS_PATH}.pkl")

# # ==========================================
# # 7. Feature importance plot
# # ==========================================
# print("\n" + "="*50)
# print("FEATURE IMPORTANCE ANALYSIS")
# print("="*50)

# try:
#     feature_importance = predictor.feature_importance(val_data, silent=True)
    
#     # Display top 20 features
#     print("\nTop 20 Most Important Features:")
#     print(feature_importance.head(20))
    
#     # Plot
#     plt.figure(figsize=(12, 8))
#     feature_importance.head(20).plot(kind='barh')
#     plt.title('Top 20 Feature Importance (AutoGluon)')
#     plt.xlabel('Importance Score')
#     plt.tight_layout()
#     plt.savefig(f"{MODEL_PATH}/feature_importance.png", dpi=150, bbox_inches='tight')
#     plt.show()
    
#     # Check if embedding features are important
#     embedding_importance = feature_importance[feature_importance.index.str.startswith('emb_')]
#     non_embedding_importance = feature_importance[~feature_importance.index.str.startswith('emb_')]
    
#     print(f"\n📊 Importance Summary:")
#     print(f"  Total embedding importance: {embedding_importance['importance'].sum():.4f}")
#     print(f"  Total non-embedding importance: {non_embedding_importance['importance'].sum():.4f}")
#     print(f"  Top non-embedding feature: {non_embedding_importance.index[0]} ({non_embedding_importance.iloc[0]['importance']:.4f})")
    
# except Exception as e:
#     print(f"⚠️ Could not compute feature importance: {e}")

# # ==========================================
# # 8. Summary for comparison with FT-Transformer
# # ==========================================
# print("\n" + "="*50)
# print("COMPARISON SUMMARY (AutoGluon)")
# print("="*50)
# # print(f"Features used: {len(available_features)} non-embedding + {emb_dim} embedding = {len(available_features) + emb_dim} total")
# print(f"Best model: {predictor.model_best}")
# print(f"Test R² (original): {test_r2_orig:.4f}")
# print(f"Validation R² (original): {val_r2_orig:.4f}")
# print(f"Training time: {leaderboard[leaderboard['model'] == predictor.model_best]['fit_time'].values[0]:.2f} seconds")
# print(f"Inference time (test): {leaderboard[leaderboard['model'] == predictor.model_best]['pred_time_test'].values[0]:.2f} seconds")

# print("\n✅ Training completed successfully!")
# print(f"Model saved at: {MODEL_PATH}")
# print(f"To load model: predictor = TabularPredictor.load('{MODEL_PATH}')")