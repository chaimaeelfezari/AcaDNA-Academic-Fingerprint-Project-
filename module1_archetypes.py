import pandas as pd
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────
df = pd.read_csv("AcaDNA_clustered.csv")
profiles = pd.read_csv("cluster_profiles.csv")

print(df["cluster_kmeans"].value_counts())
print(pd.crosstab(df["cluster_kmeans"], df["data_source"]))

# ── Archetype names (based on cluster_profiles.csv patterns) ──
archetype_names = {
    0: "The Passionate Pretender",
    1: "The Pressure Procrastinator",
    2: "The Serene Organizer",
    3: "The Exhausted Masker",
    4: "The Night Owl"
}
df["archetype"] = df["cluster_kmeans"].map(archetype_names)

# ── PCA 2D visualization ──────────────────────────────
plt.figure(figsize=(9, 6))
for cluster_id, name in archetype_names.items():
    subset = df[df["cluster_kmeans"] == cluster_id]
    plt.scatter(subset["pca1"], subset["pca2"], label=name, alpha=0.7)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Student Archetypes — AcaDNA")
plt.legend()
plt.tight_layout()
plt.savefig("archetypes_pca2d.png", dpi=150)
plt.show()
print("Saved: archetypes_pca2d.png")