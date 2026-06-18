from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
CORS(app)

# ── LOAD MODELS + DATA (une seule fois au démarrage) ──────
df = pd.read_csv('AcaDNA_final.csv')
xgb_model = joblib.load('xgboost_model.pkl')
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
             'pca1', 'pca2', 'data_source', 'mal_oriente',
             'best_match_cluster', 'match_score']
drop_cols = [c for c in drop_cols if c in df.columns]
q40_candidates = [c for c in df.columns if c.startswith('40.')]
drop_cols += q40_candidates

feature_cols = [c for c in df.columns if c not in drop_cols]
X_ref = df[feature_cols].copy()

encoders = {}
for col in X_ref.columns:
    if X_ref[col].dtype == object:
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

    # Mapping des inputs du frontend vers les colonnes du dataset
    set_val('1. How do you perceive', data.get('q1', 3))
    set_val('11. How satisfied', data.get('q11', 3))
    set_val('2. Do you tend to hide', data.get('q2', 2))
    set_val('14. Do you pretend', data.get('q14', 2))
    set_val('17. How often do you act', data.get('q17', 2))
    set_val('27. Do you delay', data.get('q27', 2))
    set_val('28. How do you describe your relationship with deadlines', data.get('q28', 2))
    set_val('36. Have you ever seriously considered', data.get('q36', 0))

    user_df = pd.DataFrame([user_vec], columns=col_names)

    # ── PRÉDICTION XGBOOST (archétype) ──────────────────────
    cluster_pred = int(xgb_model.predict(user_df)[0])
    proba = xgb_model.predict_proba(user_df)[0].tolist()

    # ── PRÉDICTION KNN (orientation) ────────────────────────
    user_scaled = scaler.transform(user_df)
    orientation_pred = int(knn_model.predict(user_scaled)[0])

    # ── DIMENSIONS POUR RADAR (0 à 1) ───────────────────────
    perf = data.get('q1', 3) / 5
    satisfaction = data.get('q11', 3) / 5
    masquage = (data.get('q2', 2) + data.get('q14', 2) + data.get('q17', 2)) / 12
    procra = (data.get('q27', 2) + data.get('q28', 2)) / 8

    return jsonify({
        'cluster': cluster_pred,
        'archetype_name': ARCHETYPE_NAMES[cluster_pred],
        'probabilities': proba,
        'mal_oriente': orientation_pred,
        'dimensions': {
            'performance': perf,
            'masking': masquage,
            'procrastination': procra,
            'satisfaction': satisfaction
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': True})

if __name__ == '__main__':
    print("Backend AcaDNA démarré sur http://localhost:5000")
    app.run(debug=True, port=5000)