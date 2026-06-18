from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from flask import send_from_directory
app = Flask(__name__)
CORS(app)

# Load all models
mask_model = joblib.load("models/mask_model.pkl")
mask_encoders = joblib.load("models/mask_encoders.pkl")
mask_feature_cols = joblib.load("models/mask_feature_cols.pkl")
archetype_model = joblib.load("models/archetype_model.pkl")
archetype_scaler = joblib.load("models/archetype_scaler.pkl")
archetype_feature_cols = joblib.load("models/archetype_feature_cols.pkl")

# Load data for success plan
df_real = pd.read_csv("AcaDNA_clustered.csv")
df_real = df_real[df_real["data_source"] == "real"].reset_index(drop=True)

archetype_names = {
    0: "The Passionate Pretender",
    1: "The Pressure Procrastinator",
    2: "The Serene Organizer",
    3: "The Exhausted Masker",
    4: "The Night Owl"
}

proc_map = {
    "plan_advance": "Disciplined",
    "pressure_motivates": "Optimizer",
    "last_moment": "Avoider",
    "sometimes_miss": "Overwhelmed",
    "often_ignore": "Overwhelmed"
}

archetype_descriptions = {
    0: "You appear outwardly fine but carry stress alone. You rarely ask for help and tend to hide how you truly feel.",
    1: "You're present but uncertain about your path. You often work last-minute and have questioned your field choice.",
    2: "You're well-matched to your field — calm, organized, and authentic. Keep that balance.",
    3: "You're at high risk of burnout. You constantly hide your struggles and feel overwhelmed. You need support.",
    4: "You perform well but stay quiet when exhausted. You're independent and introverted — that's a strength."
}

match_score_descriptions = {
    "strong": "Your answers align strongly with your archetype — you are well-placed.",
    "moderate": "Your answers partially align with your archetype — some aspects may need attention.",
    "weak": "Your answers weakly align with your archetype — you may benefit from guidance."
}

tips_per_archetype = {
    0: [
        "Try opening up to at least one trusted person about how you really feel.",
        "Asking for help is a sign of strength, not weakness.",
        "Your grades don't define your worth — it's okay to not be fine sometimes."
    ],
    1: [
        "Reflect on what genuinely interests you — a small shift in direction now saves years later.",
        "Start tasks earlier, even 10 minutes a day builds momentum.",
        "Talk to a counselor or mentor about your field — you don't have to figure it out alone."
    ],
    2: [
        "Keep organizing your time from week one — don't let work pile up.",
        "Share your methods with peers — your approach can help others.",
        "Set ambitious goals — you have the foundation to achieve them."
    ],
    3: [
        "You are not alone — many students feel exactly like you do.",
        "Start studying from the beginning, small steps every day.",
        "Prepare before class and ask questions without shame — it gets easier."
    ],
    4: [
        "Try building more social connections — they can be a real source of support.",
        "Be yourself and don't overperform just to compensate on hard days.",
        "Keep going and never give up — consistency matters more than perfection."
    ]
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ── Mask prediction ──────────────────────────────
    mask_row = {}
    for i, col in enumerate(mask_feature_cols):
        key = f"q{[1,2,14,15,16,17,18,20,21,22,23,24][i]}"
        val = data.get(key)
        if col in mask_encoders:
            le = mask_encoders[col]
            val = le.transform([val])[0] if val in le.classes_ else 0
        mask_row[col] = val

    X_mask = pd.DataFrame([mask_row])[mask_feature_cols]
    mask_pred = mask_model.predict(X_mask)[0]
    mask_conf = float(mask_model.predict_proba(X_mask)[0].max())

    # ── Archetype prediction ─────────────────────────
    arch_row = {}
    for col in archetype_feature_cols:
        arch_row[col] = data.get("arch_" + col, 0)

    X_arch = pd.DataFrame([arch_row])[archetype_feature_cols]
    X_arch_scaled = archetype_scaler.transform(X_arch)
    archetype_id = int(archetype_model.predict(X_arch_scaled)[0])

    # ── Match score ──────────────────────────────────
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    cluster_profiles = []
    X_ref = df_real.drop(columns=[c for c in df_real.columns
                                   if c in ["Horodateur","data_source",
                                            df_real.columns[40],
                                            "cluster_kmeans","cluster_dbscan",
                                            "pca1","pca2"]], errors="ignore")
    for col in X_ref.columns:
        if not pd.api.types.is_numeric_dtype(X_ref[col]):
            X_ref[col] = LabelEncoder().fit_transform(X_ref[col].astype(str))
    X_ref = X_ref.fillna(0)
    sc = StandardScaler()
    X_ref_scaled = sc.fit_transform(X_ref)

    for i in range(5):
        mask_cl = df_real["cluster_kmeans"] == i
        cluster_profiles.append(X_ref_scaled[mask_cl].mean(axis=0))
    cluster_profiles = np.array(cluster_profiles)

    student_vec = X_ref_scaled[0:1] * 0  # placeholder zeros
    match_score = float(cosine_similarity(
        cluster_profiles[archetype_id].reshape(1,-1),
        cluster_profiles[archetype_id].reshape(1,-1)
    )[0][0])

    if match_score >= 0.35:
        match_label = "strong"
    elif match_score >= 0.21:
        match_label = "moderate"
    else:
        match_label = "weak"

    # ── Procrastination ──────────────────────────────
    proc_type = proc_map.get(data.get("deadline"), "Unknown")

    # ── Success plan tips ────────────────────────────
    tips = tips_per_archetype.get(archetype_id, [])

    return jsonify({
        "mask_level": mask_pred,
        "mask_confidence": round(mask_conf, 2),
        "archetype_id": archetype_id,
        "archetype_name": archetype_names[archetype_id],
        "archetype_description": archetype_descriptions[archetype_id],
        "procrastination_type": proc_type,
        "match_label": match_label,
        "match_description": match_score_descriptions[match_label],
        "tips": tips
    })
@app.route("/")
def serve_app():
    return send_from_directory(".", "app.html")

@app.route("/<path:filename>")
def serve_file(filename):
    return send_from_directory(".", filename)
if __name__ == "__main__":
    app.run(debug=True, port=5000)