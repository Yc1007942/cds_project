import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
from scipy.stats import randint, uniform

# ==========================================
# 1. Load Moltbook Data
# ==========================================
moltbook = pd.read_pickle("../data/moltbook_with_keyword_features.pkl")
print(moltbook.head())
print(moltbook.columns)

EMBEDDING_COL = "embeddings"   # <-- CHANGE THIS to the actual column name containing the list of embeddings
TARGET_COL = "score"

RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE_FROM_TEMP = 0.50

# ==========================================
# 2. Extract and Expand Embeddings
# ==========================================
embedding_lists = moltbook[EMBEDDING_COL].values

# Check that all lists have the same length
lengths = [len(lst) for lst in embedding_lists]
if len(set(lengths)) != 1:
    raise ValueError("Embedding lists have varying lengths. Cannot expand.")

emb_dim = lengths[0]
print(f"Embedding dimension: {emb_dim}")

emb_df = pd.DataFrame(
    np.vstack(embedding_lists),
    index=moltbook.index,
    columns=[f"emb_{i}" for i in range(emb_dim)]
)
print(f"Expanded embeddings shape: {emb_df.shape}")

# ==========================================
# 3. Prepare Features and Target
# ==========================================
X_base = moltbook.drop(columns=[TARGET_COL, EMBEDDING_COL, "safe_content", "content", "id"])

# Transform target (log1p to handle skewness)
y_raw = moltbook[TARGET_COL].clip(lower=0)
y = np.log1p(y_raw)

# ==========================================
# 4. Train/Validation/Test Split
# ==========================================
X_base_train, X_base_temp, emb_train, emb_temp, y_train, y_temp = train_test_split(
    X_base, emb_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_base_val, X_base_test, emb_val, emb_test, y_val, y_test = train_test_split(
    X_base_temp, emb_temp, y_temp, test_size=VAL_SIZE_FROM_TEMP, random_state=RANDOM_STATE
)

print(f"\nTraining sizes: Base={X_base_train.shape}, Embeddings={emb_train.shape}")
print(f"Validation sizes: Base={X_base_val.shape}, Embeddings={emb_val.shape}")
print(f"Test sizes: Base={X_base_test.shape}, Embeddings={emb_test.shape}")

# ==========================================
# 5. Combine Features for Training
# ==========================================
X_train_full = pd.concat([X_base_train, emb_train], axis=1)
X_val = pd.concat([X_base_val, emb_val], axis=1)
X_test = pd.concat([X_base_test, emb_test], axis=1)

print(f"\nCombined training features shape: {X_train_full.shape}")
print(f"Combined validation features shape: {X_val.shape}")
print(f"Combined test features shape: {X_test.shape}")

# ==========================================
# 6. Baseline Model (Original Parameters)
# ==========================================
print("\n" + "="*60)
print("BASELINE MODEL")
print("="*60)

baseline_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    max_features=0.8,
    max_samples=0.8,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Cross-validation on training set
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
baseline_cv_scores = cross_val_score(baseline_model, X_train_full, y_train, 
                                      scoring='r2', cv=cv, n_jobs=-1)

print(f"\nBaseline Model - 5-fold CV R² on training data:")
print(f"  Mean: {baseline_cv_scores.mean():.4f}")
print(f"  Std:  {baseline_cv_scores.std():.4f}")
print(f"  Scores: {baseline_cv_scores}")

# Train baseline model
baseline_model.fit(X_train_full, y_train)

# Evaluate on validation and test
y_val_pred_baseline = baseline_model.predict(X_val)
y_test_pred_baseline = baseline_model.predict(X_test)

r2_val_baseline = r2_score(y_val, y_val_pred_baseline)
r2_test_baseline = r2_score(y_test, y_test_pred_baseline)

print(f"\nBaseline Model Performance:")
print(f"  Validation R²: {r2_val_baseline:.4f}")
print(f"  Test R²:       {r2_test_baseline:.4f}")

# ==========================================
# 7. Adjusted R² Function
# ==========================================
def adjusted_r2(r2, n, p):
    """Calculate adjusted R²"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

n_val, p_val = len(y_val), X_val.shape[1]
n_test, p_test = len(y_test), X_test.shape[1]

adj_r2_val_baseline = adjusted_r2(r2_val_baseline, n_val, p_val)
adj_r2_test_baseline = adjusted_r2(r2_test_baseline, n_test, p_test)

print(f"\nBaseline Model Adjusted R²:")
print(f"  Validation Adjusted R²: {adj_r2_val_baseline:.4f}")
print(f"  Test Adjusted R²:       {adj_r2_test_baseline:.4f}")

# ==========================================
# 8. Hyperparameter Tuning with RandomizedSearchCV
# ==========================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

# Parameter distributions for RandomizedSearch
param_dist = {
    'n_estimators': randint(100, 800),
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.3, 0.6),  # between 0.3 and 0.9
    'max_samples': uniform(0.6, 1.0)    # between 0.6 and 1.0
}

print("\nSearching hyperparameter space...")
print(f"  Parameters to tune: {list(param_dist.keys())}")
print(f"  Number of combinations to try: 50")

rf_tune = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

random_search = RandomizedSearchCV(
    rf_tune, 
    param_distributions=param_dist,
    n_iter=50,           # number of parameter combinations to try
    cv=5,                # 5-fold CV inside training set
    scoring='r2',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit the random search
random_search.fit(X_train_full, y_train)

print(f"\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest cross-validation R²: {random_search.best_score_:.4f}")

# Get the best model
tuned_model = random_search.best_estimator_

# ==========================================
# 9. Evaluate Tuned Model
# ==========================================
print("\n" + "="*60)
print("TUNED MODEL EVALUATION")
print("="*60)

# Cross-validation on tuned model (should match best_score_)
tuned_cv_scores = cross_val_score(tuned_model, X_train_full, y_train, 
                                   scoring='r2', cv=cv, n_jobs=-1)

print(f"\nTuned Model - 5-fold CV R² on training data:")
print(f"  Mean: {tuned_cv_scores.mean():.4f}")
print(f"  Std:  {tuned_cv_scores.std():.4f}")

# Evaluate on validation and test
y_val_pred_tuned = tuned_model.predict(X_val)
y_test_pred_tuned = tuned_model.predict(X_test)

r2_val_tuned = r2_score(y_val, y_val_pred_tuned)
r2_test_tuned = r2_score(y_test, y_test_pred_tuned)

print(f"\nTuned Model Performance:")
print(f"  Validation R²: {r2_val_tuned:.4f}")
print(f"  Test R²:       {r2_test_tuned:.4f}")

# Adjusted R² for tuned model
adj_r2_val_tuned = adjusted_r2(r2_val_tuned, n_val, p_val)
adj_r2_test_tuned = adjusted_r2(r2_test_tuned, n_test, p_test)

print(f"\nTuned Model Adjusted R²:")
print(f"  Validation Adjusted R²: {adj_r2_val_tuned:.4f}")
print(f"  Test Adjusted R²:       {adj_r2_test_tuned:.4f}")

# ==========================================
# 10. R² on Original Scale (Inverse Transform)
# ==========================================
print("\n" + "="*60)
print("PERFORMANCE ON ORIGINAL SCALE")
print("="*60)

# Inverse transform predictions and true values
y_test_orig = np.expm1(y_test)
y_test_pred_tuned_orig = np.expm1(y_test_pred_tuned)

r2_original_scale = r2_score(y_test_orig, y_test_pred_tuned_orig)
print(f"R² on original score scale (test set): {r2_original_scale:.4f}")

# ==========================================
# 11. Feature Importances (Top 20)
# ==========================================
print("\n" + "="*60)
print("FEATURE IMPORTANCES")
print("="*60)

importances = tuned_model.feature_importances_
feat_names = X_train_full.columns
df_imp = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
df_imp = df_imp.sort_values('Importance', ascending=False).head(20)

print("\nTop 20 Features:")
for idx, row in df_imp.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Plot feature importances
plt.figure(figsize=(10, 8))
plt.barh(df_imp['Feature'][::-1], df_imp['Importance'][::-1], color='lightcoral')
plt.xlabel('Importance Score')
plt.title('Top 20 Features: Tuned Random Forest')
plt.tight_layout()
plt.savefig('feature_importances_tuned.png', dpi=150)
plt.show()

# ==========================================
# 12. Performance Comparison Summary
# ==========================================
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

summary_data = {
    'Metric': ['CV R² (mean)', 'CV R² (std)', 'Validation R²', 'Test R²', 'Validation Adj. R²', 'Test Adj. R²'],
    'Baseline': [
        f"{baseline_cv_scores.mean():.4f}",
        f"{baseline_cv_scores.std():.4f}",
        f"{r2_val_baseline:.4f}",
        f"{r2_test_baseline:.4f}",
        f"{adj_r2_val_baseline:.4f}",
        f"{adj_r2_test_baseline:.4f}"
    ],
    'Tuned': [
        f"{tuned_cv_scores.mean():.4f}",
        f"{tuned_cv_scores.std():.4f}",
        f"{r2_val_tuned:.4f}",
        f"{r2_test_tuned:.4f}",
        f"{adj_r2_val_tuned:.4f}",
        f"{adj_r2_test_tuned:.4f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Calculate improvement
improvement = (r2_test_tuned - r2_test_baseline) * 100
print(f"\nTest R² Improvement: {improvement:+.2f} percentage points")

# ==========================================
# 13. Save Models
# ==========================================
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

joblib.dump(baseline_model, 'random_forest_baseline.joblib')
joblib.dump(tuned_model, 'random_forest_tuned.joblib')
joblib.dump(random_search, 'random_forest_search.joblib')

print("Models saved:")
print("  - random_forest_baseline.joblib")
print("  - random_forest_tuned.joblib")
print("  - random_forest_search.joblib")

# Optional: Save feature importances
df_imp.to_csv('feature_importances.csv', index=False)
print("  - feature_importances.csv")

print("\n" + "="*60)
print("COMPLETE! All analyses finished successfully.")
print("="*60)