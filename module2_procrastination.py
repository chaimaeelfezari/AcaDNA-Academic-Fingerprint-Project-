import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("AcaDNA_clustered.csv")
df = df[df["data_source"] == "real"].reset_index(drop=True)
print("Real rows:", len(df))

# ── Target: procrastination type ─────────────────────
proc_target_col = df.columns[28]
proc_map = {
    "I plan well in advance / أخطط مسبقاً جيداً": "Disciplined",
    "Pressure motivates me / الضغط يحفزني": "Optimizer",
    "I meet them but at the last moment / ألتزم بها لكن في اللحظة الأخيرة": "Avoider",
    "I sometimes miss them / أتجاوزها أحياناً": "Overwhelmed",
    "I often ignore them / أتجاهلها غالباً": "Overwhelmed"
}
df["procrastination_type"] = df[proc_target_col].map(proc_map)
print(df["procrastination_type"].value_counts())

# ── Features ─────────────────────────────────────────
feature_cols = [df.columns[25], df.columns[26], df.columns[27], df.columns[29],
                 df.columns[30], df.columns[31], df.columns[32], df.columns[33],
                 df.columns[34]]
X = df[feature_cols]

for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

y_le = LabelEncoder()
y = y_le.fit_transform(df["procrastination_type"])
print(dict(zip(y_le.classes_, range(len(y_le.classes_)))))

# ── Train/test split + balanced single model ─────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="mlogloss")
model.fit(X_train, y_train, sample_weight=sample_weights)
print(classification_report(y_test, model.predict(X_test), target_names=y_le.classes_))

# ── Balanced cross-validation (manual, correct) ──────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
for train_idx, test_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    w = compute_sample_weight(class_weight="balanced", y=y_tr)
    m = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="mlogloss")
    m.fit(X_tr, y_tr, sample_weight=w)
    fold_scores.append(m.score(X_te, y_te))
print("Balanced CV scores:", fold_scores)
print("Balanced CV average:", np.mean(fold_scores))

# ── Hyperparameter search ─────────────────────────────
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2]
}
grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="mlogloss"),
    param_grid, cv=5, scoring="accuracy"
)
grid.fit(X, y)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

best_model = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                            random_state=42, eval_metric="mlogloss")
y_pred_cv = cross_val_predict(best_model, X, y, cv=5)
print(classification_report(y, y_pred_cv, target_names=y_le.classes_))
print(confusion_matrix(y, y_pred_cv))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y,
    y_pred_cv,
    display_labels=y_le.classes_
)

plt.title("Procrastination Type Prediction")
plt.tight_layout()
plt.savefig("procrastination_confusion_matrix.png", dpi=150)
plt.show()