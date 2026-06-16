import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('AcaDNA_clustered.csv')

# ── 1. FEATURES ───────────────────────────────────────────
drop_cols = ['Horodateur', 'cluster_kmeans', 'cluster_dbscan',
             'pca1', 'pca2', 'data_source', df.columns[40]]

X = df.drop(columns=drop_cols)
for col in X.columns:
    if X[col].dtype == object:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 2. TARGET = satisfaction filière (Q11) ────────────────
# Q35 : "Do your natural strengths match your field?"
q35 = [c for c in df.columns if '35.' in c][0]
q36 = [c for c in df.columns if '36.' in c][0]

# Créer label orientation : 0=bien orienté, 1=mal orienté
df['mal_oriente'] = 0
df.loc[df[q35].str.contains('No|Somewhat', na=False), 'mal_oriente'] = 1
df.loc[df[q36].str.contains('Yes', na=False), 'mal_oriente'] = 1

y = df['mal_oriente']
print(f"Mal orientés: {y.sum()} / {len(y)}")

# ── 3. KNN ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Chercher meilleur K
scores = []
for k in range(3, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append((k, knn.score(X_test, y_test)))
    print(f"K={k} | Accuracy={knn.score(X_test, y_test):.3f}")

best_k = max(scores, key=lambda x: x[1])[0]
print(f"\nMeilleur K = {best_k}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
print(classification_report(y_test, knn_final.predict(X_test)))

# ── 4. COSINE SIMILARITY ──────────────────────────────────
# Profil moyen de chaque cluster (archétype)
cluster_profiles = []
for i in range(5):
    mask = df['cluster_kmeans'] == i
    cluster_profiles.append(X_scaled[mask].mean(axis=0))
cluster_profiles = np.array(cluster_profiles)

# Similarité de chaque étudiant avec chaque archétype
sim_matrix = cosine_similarity(X_scaled, cluster_profiles)
df['best_match_cluster'] = sim_matrix.argmax(axis=1)
df['match_score'] = sim_matrix.max(axis=1)

print("\n=== Cosine Similarity — match moyen par cluster ===")
print(df.groupby('cluster_kmeans')['match_score'].mean())

# ── 5. PLOT ───────────────────────────────────────────────
# KNN accuracy vs K
ks, accs = zip(*scores)
plt.figure(figsize=(8, 4))
plt.plot(ks, accs, 'bo-')
plt.title('KNN — Accuracy vs K')
plt.xlabel('K'); plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('knn_accuracy.png', dpi=150)
plt.close()

# Distribution mal orientés par cluster
fig, ax = plt.subplots(figsize=(8, 4))
df.groupby('cluster_kmeans')['mal_oriente'].mean().plot(
    kind='bar', ax=ax, color='coral')
ax.set_title('% Mal orientés par archétype')
ax.set_xlabel('Cluster'); ax.set_ylabel('Proportion')
ax.set_xticklabels([f'C{i}' for i in range(5)], rotation=0)
plt.tight_layout()
plt.savefig('orientation_par_cluster.png', dpi=150)
plt.close()

# Save
df.to_csv('AcaDNA_final.csv', index=False)
print("\nSaved: AcaDNA_final.csv, knn_accuracy.png, orientation_par_cluster.png ✅")