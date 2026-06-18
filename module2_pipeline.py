import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("AcaDNA_clustered.csv")

# ── Use only real responses (synthetic ones dilute signal) ──
df = df[df["data_source"] == "real"].reset_index(drop=True)
print("Real rows:", len(df))

# ── Target: mask level (binary) ──────────────────────
target_col = df.columns[19]
mask_map = {
    "No, I am always myself / لا، أنا نفسي دائماً": "No_mask",
    "A little / قليلاً": "No_mask",
    "Sometimes yes / أحياناً نعم": "Mask",
    "Often / غالباً": "Mask",
    "Almost all the time / تقريباً طوال الوقت": "Mask"
}
df["mask_level"] = df[target_col].map(mask_map)
print(df["mask_level"].value_counts())

# ── Features ─────────────────────────────────────────
feature_cols = [df.columns[1], df.columns[2], df.columns[14], df.columns[15],
                 df.columns[16], df.columns[17], df.columns[18], df.columns[20],
                 df.columns[21], df.columns[22], df.columns[23], df.columns[24]]
X = df[feature_cols]
y = df["mask_level"]

for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Hyperparameter search ────────────────────────────
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [1, 2, 4]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid, cv=5, scoring="accuracy"
)
grid.fit(X, y)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

best_rf = grid.best_estimator_
best_rf.fit(X_train, y_train)
print(classification_report(y_test, best_rf.predict(X_test)))

scores = cross_val_score(best_rf, X, y, cv=5, scoring="accuracy")
print("Final CV scores:", scores)
print("Final average:", scores.mean())

importances = pd.Series(best_rf.feature_importances_, index=[c.split('.')[0] for c in feature_cols])
print(importances.sort_values(ascending=False))

# ── Compare model types ──────────────────────────────
y_num = y.map({"No_mask": 0, "Mask": 1})

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=1,
                                            random_state=42, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}
for name, m in models.items():
    s = cross_val_score(m, X, y_num, cv=5, scoring="accuracy")
    print(name, "-> mean accuracy:", s.mean())

# ── Check it's not just guessing the majority class ──
rf_final = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=1,
                                   random_state=42, class_weight="balanced")
y_pred_cv = cross_val_predict(rf_final, X, y, cv=5)
print(classification_report(y, y_pred_cv))
print(confusion_matrix(y, y_pred_cv, labels=["No_mask", "Mask"]))

# ── SMOTE: oversample the Mask class properly, training-fold only ──
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

smote_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42, k_neighbors=5)),
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=1, random_state=42))
])
smote_scores = cross_val_score(smote_pipeline, X, y, cv=5, scoring="accuracy")
print("SMOTE + RF CV scores:", smote_scores)
print("SMOTE + RF average:", smote_scores.mean())

import matplotlib.pyplot as plt

importances.sort_values().plot(kind="barh", figsize=(8,5))
plt.title("Mask Detection - Feature Importance")
plt.tight_layout()
plt.savefig("mask_feature_importance.png", dpi=150)
plt.show()