import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
from sklearn.neighbors import NearestNeighbors
import numpy as np

warnings.filterwarnings('ignore')

df = pd.read_csv("AcaDNA_augmented_300_fixed.csv")

# ── 1. DROP colonnes inutiles ──────────────────────────────
df_clean = df.drop(columns=['Horodateur', 'data_source',
    df.columns[40]])  # Q40 = texte libre

# -- 2. Mappings ordinaux explicites
ordinal_maps = {
    'Never':0, 'Rarely':1, 'Sometimes':2, 'Often':3, 'Always':4,
    'A little':1, 'Limited':1, 'Active':3, 'Very active':4,
    '1–2h':1, '2–4h':2, '4–6h':3, '6h+':4,
    '5–6h':1, '6–7h':2, '7–8h':3, '8h+':4,
}

def smart_encode(val):
    val = str(val)
    for k,v in ordinal_maps.items():
        if k.lower() in val.lower():
            return v
    return None

for col in df_clean.columns:
    if df_clean[col].dtype == object:
        mapped = df_clean[col].apply(smart_encode)
        if mapped.notna().sum() > len(df_clean)*0.5:
            df_clean[col] = mapped.fillna(mapped.median())
        else:
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

# ── 3. SCALE ──────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(df_clean)

# ── 4. ELBOW + SILHOUETTE pour choisir K ─────────────────
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))
    print(f"K={k} | Inertia={km.inertia_:.0f} | Silhouette={silhouette_score(X,labels):.3f}")

# ── 5. PLOT elbow ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(K_range, inertias, 'bo-')
ax1.set_title('Elbow Method'); ax1.set_xlabel('K'); ax1.set_ylabel('Inertia')
ax2.plot(K_range, sil_scores, 'ro-')
ax2.set_title('Silhouette Score'); ax2.set_xlabel('K')
plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150)
print("\nSaved: elbow_silhouette.png")

# ── 6. K-MEANS avec meilleur K ───────────────────────────
best_k = 5  # forcé : silhouette plat → choix sémantique justifié
print(f"\nMeilleur K = {best_k}")
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster_kmeans'] = km_final.fit_predict(X)

# ── 7. DBSCAN ───────────────────────────────────────────
nbrs = NearestNeighbors(n_neighbors=8).fit(X)
distances, _ = nbrs.kneighbors(X)
eps_auto = np.percentile(distances[:, -1], 90)
print(f"eps auto = {eps_auto:.3f}")

db = DBSCAN(eps=eps_auto, min_samples=8)
df['cluster_dbscan'] = db.fit_predict(X)
# ── 8. PCA 2D + PLOT ──────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
df['pca1'] = X_2d[:, 0]
df['pca2'] = X_2d[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
scatter1 = ax1.scatter(X_2d[:,0], X_2d[:,1], c=df['cluster_kmeans'], cmap='tab10', alpha=0.7)
ax1.set_title(f'K-Means (K={best_k})'); plt.colorbar(scatter1, ax=ax1)
scatter2 = ax2.scatter(X_2d[:,0], X_2d[:,1], c=df['cluster_dbscan'], cmap='tab10', alpha=0.7)
ax2.set_title('DBSCAN'); plt.colorbar(scatter2, ax=ax2)
plt.tight_layout()
plt.savefig('clusters_pca2d.png', dpi=150)
print("Saved: clusters_pca2d.png")

# ── 9. SAVE ───────────────────────────────────────────────
df.to_csv('AcaDNA_clustered.csv', index=False)
print("Saved: AcaDNA_clustered.csv")
print(f"\nDistribution K-Means:\n{df['cluster_kmeans'].value_counts().sort_index()}")
# Profil moyen par cluster
df_orig = pd.read_csv('AcaDNA_augmented_300_fixed.csv')
df_orig['cluster'] = df['cluster_kmeans']
profile = df_orig.groupby('cluster').agg(lambda x: x.mode()[0]).T
profile.to_csv('cluster_profiles.csv')
print(profile)