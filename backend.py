from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app)

# ── LOAD MODELS + DATA ─────────────────────────────────────
df = pd.read_csv('AcaDNA_with_scores.csv')
model_procra = joblib.load('xgboost_procrastination.pkl')
model_masq = joblib.load('xgboost_masquage.pkl')
knn_model = joblib.load('knn_model.pkl')

ARCHETYPE_NAMES = {
    0: "The Night Owl",
    1: "The Passionate Pretender",
    2: "The Serene Organizer",
    3: "The Pressure Procrastinator",
    4: "The Exhausted Masker"
}

# ── PREPARE FEATURE REFERENCE (même pipeline que l'entraînement) ──
drop_cols = ['Horodateur', 'cluster_kmeans', 'cluster_dbscan',
             'pca1', 'pca2', 'data_source', 'score_masquage',
             'score_procrastination', 'mal_oriente',
             'best_match_cluster', 'match_score']
drop_cols = [c for c in drop_cols if c in df.columns]
q40_candidates = [c for c in df.columns if c.startswith('40.')]
drop_cols += q40_candidates

feature_cols = [c for c in df.columns if c not in drop_cols]
X_ref = df[feature_cols].copy()

# Mapping ordinal correct (au lieu de LabelEncoder alphabetique qui casse l'ordre)
FREQ_MAP = {
    'Never':0, 'Rarely':1, 'Sometimes':2, 'Often':3, 'Always':4
}

def ordinal_encode(series):
    def map_val(x):
        x = str(x)
        for k, v in FREQ_MAP.items():
            if k in x:
                return v
        return None
    mapped = series.apply(map_val)
    if mapped.notna().sum() > len(series) * 0.5:
        return mapped.fillna(mapped.median())
    return None

encoders = {}
for col in X_ref.columns:
    if X_ref[col].dtype == object:
        ordinal = ordinal_encode(X_ref[col])
        if ordinal is not None:
            X_ref[col] = ordinal
        else:
            le = LabelEncoder()
            X_ref[col] = le.fit_transform(X_ref[col].astype(str))
            encoders[col] = le
X_ref = X_ref.fillna(0)

scaler = StandardScaler()
X_scaled_ref = scaler.fit_transform(X_ref)

col_names = X_ref.columns.tolist()
median_vector = X_ref.median().values.copy()

def find_col(keyword):
    matches = [c for c in col_names if keyword in c]
    return matches[0] if matches else None

def archetype_from_scores(procra, masq, perf, satisfaction):
    """Détermine l'archétype le plus probable à partir des scores réels du modèle"""
    scores = np.zeros(5)
    scores[0] = perf * 0.35 + (1 - masq) * 0.35 + (1 - procra) * 0.15 + (1 - satisfaction) * 0.15
    scores[1] = satisfaction * 0.3 + masq * 0.5 + perf * 0.1 + (1 - procra) * 0.1
    scores[2] = perf * 0.25 + (1 - procra) * 0.35 + satisfaction * 0.35 + (1 - masq) * 0.05
    scores[3] = procra * 0.55 + (1 - perf) * 0.45
    scores[4] = masq * 0.4 + (1 - satisfaction) * 0.4 + (1 - perf) * 0.2

    T = 0.12
    exp = np.exp((scores - scores.max()) / T)
    proba = exp / exp.sum()
    return int(np.argmax(proba)), proba.tolist()

# ── ROUTES ─────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    user_vec = median_vector.copy()

    def set_val(keyword, value):
        col = find_col(keyword)
        if col:
            idx = col_names.index(col)
            user_vec[idx] = value

    set_val('1. How do you perceive', data.get('q1', 3))
    set_val('11. How satisfied', data.get('q11', 3))

    # Masquage : propager la même intensité a toutes les colonnes liees
    masq_val = data.get('q2', 2)
    for kw in ['2. Do you tend to hide', '14. Do you pretend', '15. Do you feel that your good grades',
               '16. Do you avoid asking', '17. How often do you act', '18. Do you smile',
               '19. Do you feel like you are playing', '21. Have you ever felt', '22. How often do you feel that your real level']:
        set_val(kw, masq_val)

    # Procrastination : propager a toutes les colonnes liees
    procra_val = data.get('q27', 2)
    set_val('25. When you make a study plan', 4 - procra_val)  # inverse: suivre le plan = oppose de procrastiner
    set_val('27. Do you delay', procra_val)
    set_val('28. How do you describe your relationship with deadlines', data.get('q28', 2))
    set_val('29. How often do you use', procra_val)
    set_val('30. Does fear of doing badly', procra_val)

    set_val('36. Have you ever seriously considered', data.get('q36', 0))

    user_df = pd.DataFrame([user_vec], columns=col_names)

    # ── VRAIES PREDICTIONS XGBOOST (regression) ─────────────
    procra_score = float(model_procra.predict(user_df)[0])  # 0 a 4
    masq_score = float(model_masq.predict(user_df)[0])      # 0 a 4

    # Normaliser 0-4 vers 0-1 pour le radar
    procra_norm = np.clip(procra_score / 4, 0, 1)
    masq_norm = np.clip(masq_score / 4, 0, 1)
    perf_norm = data.get('q1', 3) / 5
    satisfaction_norm = data.get('q11', 3) / 5

    # ── ARCHETYPE DERIVE DES VRAIS SCORES ML ────────────────
    cluster_pred, proba = archetype_from_scores(procra_norm, masq_norm, perf_norm, satisfaction_norm)

    # ── PREDICTION KNN (orientation) ────────────────────────
    user_scaled = scaler.transform(user_df)
    orientation_pred = int(knn_model.predict(user_scaled)[0])

    return jsonify({
        'cluster': cluster_pred,
        'archetype_name': ARCHETYPE_NAMES[cluster_pred],
        'probabilities': proba,
        'mal_oriente': orientation_pred,
        'procrastination_score': procra_score,
        'masquage_score': masq_score,
        'dimensions': {
            'performance': perf_norm,
            'masking': masq_norm,
            'procrastination': procra_norm,
            'satisfaction': satisfaction_norm
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': True})

if __name__ == '__main__':
    print("Backend AcaDNA démarré sur http://localhost:5000")
    app.run(debug=True, port=5000)