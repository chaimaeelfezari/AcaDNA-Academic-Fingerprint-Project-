import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ── LOAD SON MODELE EXACT ──────────────────────────────────
mask_model = joblib.load('models/mask_model.pkl')
mask_encoders = joblib.load('models/mask_encoders.pkl')
mask_feature_cols = joblib.load('models/mask_feature_cols.pkl')

df = pd.read_csv('AcaDNA_clustered.csv')
df = df[df['data_source'] == 'real'].reset_index(drop=True)  # meme filtre que lui

# ── REPRODUIRE EXACTEMENT SON ENCODAGE ─────────────────────
X = df[mask_feature_cols].copy()
for col in mask_feature_cols:
    if col in mask_encoders:
        le = mask_encoders[col]
        X[col] = le.transform(X[col].astype(str))

print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
print("Colonnes:", [c.split('.')[0] for c in mask_feature_cols])

# ── SHAP SUR SON MODELE (RandomForest) ─────────────────────
explainer = shap.TreeExplainer(mask_model)
shap_values_raw = explainer.shap_values(X)

# RandomForest binaire -> shap_values est une liste [class0, class1]
print("Shape brute:", shap_values_raw.shape)
sv_mask_class = shap_values_raw[:, :, 1]
print("Shape apres slicing:", sv_mask_class.shape)

plt.figure()
shap.summary_plot(sv_mask_class, X.values, feature_names=[c.split('.')[0]+'.'+c.split('|')[0].split('.',1)[1][:40] for c in mask_feature_cols], show=False, max_display=12)
plt.title("SHAP — Masquage (modele final equipe)")
plt.tight_layout()
plt.savefig('shap_masquage_FINAL.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_masquage_FINAL.png")

# Bar plot importance globale
shap.summary_plot(sv_mask_class, X, plot_type='bar', show=False, max_display=12)
plt.title("SHAP — Importance globale des features (Masquage)")
plt.tight_layout()
plt.savefig('shap_masquage_bar_FINAL.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_masquage_bar_FINAL.png")