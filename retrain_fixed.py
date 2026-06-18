import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('AcaDNA_clustered.csv')

# ── 1. SCORES CIBLES (deja correct, garde la meme logique) ─
freq_map = {'Never':0,'Rarely':1,'Sometimes':2,'Often':3,'Always':4}

def map_freq(col):
    return df[col].apply(lambda x: next((v for k,v in freq_map.items() if k in str(x)), 2))

masquage_cols = [c for c in df.columns if any(c.startswith(p) for p in
    ['2.','14.','15.','16.','17.','18.','19.','21.','22.'])]
procra_cols = [c for c in df.columns if any(c.startswith(p) for p in
    ['25.','27.','28.','29.','30.'])]

df['score_masquage'] = sum(map_freq(c) for c in masquage_cols) / len(masquage_cols)
df['score_procrastination'] = sum(map_freq(c) for c in procra_cols) / len(procra_cols)

print(f"Score masquage: min={df['score_masquage'].min():.2f} max={df['score_masquage'].max():.2f}")
print(f"Score procrastination: min={df['score_procrastination'].min():.2f} max={df['score_procrastination'].max():.2f}")

# ── 2. FEATURES avec ENCODAGE ORDINAL CORRECT ──────────────
drop_cols = ['Horodateur','cluster_kmeans','cluster_dbscan','pca1','pca2',
             'data_source','score_masquage','score_procrastination']
drop_cols += [c for c in df.columns if c.startswith('40.')]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])

def ordinal_encode(series):
    def map_val(x):
        x = str(x)
        for k, v in freq_map.items():
            if k in x:
                return v
        return None
    mapped = series.apply(map_val)
    if mapped.notna().sum() > len(series) * 0.5:
        return mapped.fillna(mapped.median())
    return None

for col in X.columns:
    if X[col].dtype == object:
        ordinal = ordinal_encode(X[col])
        if ordinal is not None:
            X[col] = ordinal
        else:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(0)

print(f"\nFeatures: {X.shape[1]} | Samples: {X.shape[0]}")

# ── 3. MODELE PROCRASTINATION ──────────────────────────────
y_procra = df['score_procrastination']
X_train, X_test, y_train, y_test = train_test_split(X, y_procra, test_size=0.2, random_state=42)

model_procra = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
model_procra.fit(X_train, y_train)
pred = model_procra.predict(X_test)
print(f"\n=== Modele Procrastination (encodage ordinal corrige) ===")
print(f"R2 = {r2_score(y_test, pred):.3f} | MAE = {mean_absolute_error(y_test, pred):.3f}")

# ── 4. MODELE MASQUAGE ─────────────────────────────────────
y_masq = df['score_masquage']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_masq, test_size=0.2, random_state=42)

model_masq = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
model_masq.fit(X_train2, y_train2)
pred2 = model_masq.predict(X_test2)
print(f"\n=== Modele Masquage (encodage ordinal corrige) ===")
print(f"R2 = {r2_score(y_test2, pred2):.3f} | MAE = {mean_absolute_error(y_test2, pred2):.3f}")

# ── 5. VERIFICATION SENS (sanity check) ────────────────────
sample_high = X.iloc[[0]].copy()
for c in masquage_cols:
    sample_high[c] = 4
sample_low = X.iloc[[0]].copy()
for c in masquage_cols:
    sample_low[c] = 0

pred_high = model_masq.predict(sample_high)[0]
pred_low = model_masq.predict(sample_low)[0]
print(f"\n=== SANITY CHECK MASQUAGE ===")
print(f"Avec toutes reponses = 4 (Always) -> score predit = {pred_high:.2f}")
print(f"Avec toutes reponses = 0 (Never)  -> score predit = {pred_low:.2f}")
print("OK si high > low" if pred_high > pred_low else "ERREUR encore inverse")

# ── 6. SHAP ─────────────────────────────────────────────────
explainer_procra = shap.TreeExplainer(model_procra)
shap_values_procra = explainer_procra(X)
shap.summary_plot(shap_values_procra, X, show=False, max_display=12)
plt.title("SHAP - Facteurs de Procrastination")
plt.tight_layout()
plt.savefig('shap_procrastination.png', dpi=150, bbox_inches='tight')
plt.close()

explainer_masq = shap.TreeExplainer(model_masq)
shap_values_masq = explainer_masq(X)
shap.summary_plot(shap_values_masq, X, show=False, max_display=12)
plt.title("SHAP - Facteurs de Masquage Academique")
plt.tight_layout()
plt.savefig('shap_masquage.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: shap_procrastination.png, shap_masquage.png")

# ── 7. SAVE ─────────────────────────────────────────────────
joblib.dump(model_procra, 'xgboost_procrastination.pkl')
joblib.dump(model_masq, 'xgboost_masquage.pkl')
df.to_csv('AcaDNA_with_scores.csv', index=False)
print("Saved models (encodage corrige) ✅")