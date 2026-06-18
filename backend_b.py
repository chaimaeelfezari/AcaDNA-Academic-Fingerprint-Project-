from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

mask_model = joblib.load("models/mask_model.pkl")
mask_encoders = joblib.load("models/mask_encoders.pkl")
mask_feature_cols = joblib.load("models/mask_feature_cols.pkl")

proc_map = {
    "plan_advance": "Disciplined",
    "pressure_motivates": "Optimizer",
    "last_moment": "Avoider",
    "sometimes_miss": "Overwhelmed",
    "often_ignore": "Overwhelmed"
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

simple_to_full = None

def get_simple_to_full():
    global simple_to_full
    if simple_to_full is None:
        simple_to_full = {
            "q1": mask_feature_cols[0], "q2": mask_feature_cols[1],
            "q14": mask_feature_cols[2], "q15": mask_feature_cols[3],
            "q16": mask_feature_cols[4], "q17": mask_feature_cols[5],
            "q18": mask_feature_cols[6], "q20": mask_feature_cols[7],
            "q21": mask_feature_cols[8], "q22": mask_feature_cols[9],
            "q23": mask_feature_cols[10], "q24": mask_feature_cols[11],
        }
    return simple_to_full

@app.route("/")
def serve_app():
    return send_from_directory(".", "app.html")

@app.route("/<path:filename>")
def serve_file(filename):
    return send_from_directory(".", filename)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    mapping = get_simple_to_full()

    row = {}
    for simple_key, full_col in mapping.items():
        val = data.get(simple_key)
        if full_col in mask_encoders:
            le = mask_encoders[full_col]
            val = le.transform([val])[0] if val in le.classes_ else 0
        row[full_col] = val

    X_new = pd.DataFrame([row])[mask_feature_cols]
    mask_pred = mask_model.predict(X_new)[0]
    mask_conf = float(mask_model.predict_proba(X_new)[0].max())

    proc_type = proc_map.get(data.get("deadline"), "Unknown")

    # pick tips based on mask level
    if mask_pred == "Mask":
        tips = tips_per_archetype[3]  # Exhausted Masker tips
    else:
        tips = tips_per_archetype[2]  # Serene Organizer tips

    return jsonify({
        "mask_level": mask_pred,
        "mask_confidence": round(mask_conf, 2),
        "procrastination_type": proc_type,
        "tips": tips
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)