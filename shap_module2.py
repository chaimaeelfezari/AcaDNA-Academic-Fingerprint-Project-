import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('AcaDNA_clustered.csv')

# ── 1. FEATURES : encoder toutes colonnes objet ───────────
drop_cols = ['Horodateur', 'cluster_kmeans', 'cluster_dbscan',
             'pca1', 'pca2', 'data_source']
# drop Q40 texte libre (colonne 40)
q40 = df.columns[40]
drop_cols.append(q40)

X = df.drop(columns=drop_cols)
y = df['cluster_kmeans']

le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == object:
        X[col] = le.fit_transform(X[col].astype(str))

X = X.fillna(0)
print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")

# ── 2. XGBOOST ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(n_estimators=100, max_depth=4,
                      random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

# ── 3. SHAP (fix multiclass) ──────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)  # nouveau format Explanation object

# Summary bar global
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary.png")

# Beeswarm par cluster
for i in range(5):
    shap.summary_plot(shap_values[:, :, i], X, show=False, max_display=10)
    plt.title(f"Cluster {i}")
    plt.tight_layout()
    plt.savefig(f'shap_cluster_{i}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: shap_cluster_{i}.png")