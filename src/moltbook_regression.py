import pandas as pd
import numpy as np
# import umap
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import os
# 1. Load Moltbook Data
moltbook = pd.read_pickle("../data/processed_v1_5_9_hf_pure_full.pkl")
print(moltbook.head())
print(moltbook.columns)
# df = moltbook[['comment_existence', 'avg_early_sentiment',
#        'max_early_sentiment', 'min_early_sentiment', 'hour', 'ttr', 'hapax',
#        'stopword_ratio', 'burstiness', 'punctuation_density', 'hedging_score',
#        'self_reference_rate', 'forum_philosophy', 'forum_technology',
#        'forum_todayilearned']]

EMBEDDING_COL = "embeddings"   # <-- CHANGE THIS to the actual column name containing the list of embeddings
TARGET_COL = "score"

RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE_FROM_TEMP = 0.50

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

X_base = moltbook.drop(columns=[TARGET_COL, EMBEDDING_COL,"safe_content","content","id"])


# y = moltbook[TARGET_COL]

y_raw = moltbook[TARGET_COL].clip(lower=0)
y = np.log1p(y_raw)


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
# 5. Model: Base Features + Raw Embeddings
# ==========================================
print("\nTraining Random Forest (Base + Raw Embeddings)...")
X_train = pd.concat([X_base_train, emb_train], axis=1)
X_val   = pd.concat([X_base_val, emb_val], axis=1)
X_test  = pd.concat([X_base_test, emb_test], axis=1)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    max_features=0.8,
    max_samples=0.8,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
model.fit(X_train, y_train)

# Evaluate
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Validation R²: {r2_val:.4f}")
print(f"Test R²: {r2_test:.4f}")

# ==========================================
# 6. Feature Importances (Top 20)
# ==========================================
importances = model.feature_importances_
feat_names = X_train.columns
df_imp = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
df_imp = df_imp.sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(df_imp['Feature'][::-1], df_imp['Importance'][::-1], color='lightcoral')
plt.xlabel('Importance Score')
plt.title('Top 20 Features: Base + Raw Embeddings')
plt.tight_layout()
# plt.savefig('feature_importances.png', dpi=150)
plt.show()

joblib.dump(model, 'random_forest_model.joblib')
