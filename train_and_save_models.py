import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

df = pd.read_csv("AcaDNA_clustered.csv")
df = df[df["data_source"] == "real"].reset_index(drop=True)

mask_map = {
    "No, I am always myself / لا، أنا نفسي دائماً": "No_mask",
    "A little / قليلاً": "No_mask",
    "Sometimes yes / أحياناً نعم": "Mask",
    "Often / غالباً": "Mask",
    "Almost all the time / تقريباً طوال الوقت": "Mask"
}
df["mask_level"] = df[df.columns[19]].map(mask_map)

mask_feature_cols = [df.columns[1], df.columns[2], df.columns[14], df.columns[15],
                      df.columns[16], df.columns[17], df.columns[18], df.columns[20],
                      df.columns[21], df.columns[22], df.columns[23], df.columns[24]]

X = df[mask_feature_cols].copy()
y = df["mask_level"]

encoders = {}
for col in mask_feature_cols:
    if not pd.api.types.is_numeric_dtype(X[col]):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

rf = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=1,
                             random_state=42, class_weight="balanced")
rf.fit(X, y)

joblib.dump(rf, "models/mask_model.pkl")
joblib.dump(encoders, "models/mask_encoders.pkl")
joblib.dump(mask_feature_cols, "models/mask_feature_cols.pkl")

print("Saved mask model + encoders")
print("Feature columns (in order):", [c.split('.')[0] for c in mask_feature_cols])


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

drop_cols_km = ["Horodateur", "data_source", df.columns[40],
                "cluster_kmeans", "cluster_dbscan", "pca1", "pca2"]
X_km = df.drop(columns=drop_cols_km)

for col in X_km.columns:
    if not pd.api.types.is_numeric_dtype(X_km[col]):
        X_km[col] = LabelEncoder().fit_transform(X_km[col].astype(str))
X_km = X_km.fillna(0)

scaler = StandardScaler()
X_km_scaled = scaler.fit_transform(X_km)

from sklearn.neighbors import KNeighborsClassifier
knn_arch = KNeighborsClassifier(n_neighbors=5)
knn_arch.fit(X_km_scaled, df["cluster_kmeans"])

joblib.dump(knn_arch, "models/archetype_model.pkl")
joblib.dump(scaler, "models/archetype_scaler.pkl")
joblib.dump(X_km.columns.tolist(), "models/archetype_feature_cols.pkl")

print("Saved archetype model")