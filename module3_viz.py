import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AcaDNA_final.csv")
df = df[df["data_source"] == "real"].reset_index(drop=True)

print(df["mal_oriente"].value_counts())
print(df["match_score"].describe())
print(df["best_match_cluster"].value_counts())

threshold = df["match_score"].quantile(0.25)
df["needs_attention"] = df["match_score"] <= threshold
print(f"Threshold: {threshold:.3f}")
print(df["needs_attention"].value_counts())

archetype_names = {
    0: "The Passionate Pretender",
    1: "The Pressure Procrastinator",
    2: "The Serene Organizer",
    3: "The Exhausted Masker",
    4: "The Night Owl"
}

# ── Visualization ─────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.hist(df[~df["needs_attention"]]["match_score"], bins=20, alpha=0.7, label="Good match", color="seagreen")
plt.hist(df[df["needs_attention"]]["match_score"], bins=20, alpha=0.7, label="Needs attention", color="indianred")
plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})")
plt.xlabel("Match score (cosine similarity to best-fit archetype)")
plt.ylabel("Number of students")
plt.title("Major Fit Distribution — AcaDNA")
plt.legend()
plt.tight_layout()
plt.savefig("match_score_distribution.png", dpi=150)
plt.show()
print("Saved: match_score_distribution.png")

# ── Concrete cases (only real archetype-change suggestions) ──
q12 = df.columns[12]  # why chose field
q36 = df.columns[36]  # considered changing
q39 = df.columns[39]  # would choose same field again

cases = df[(df["needs_attention"]) & (df["best_match_cluster"] != df["cluster_kmeans"])]
cases = cases.sort_values("match_score").head(3)

for idx, row in cases.iterrows():
    print("─" * 50)
    print(f"Current archetype:   {archetype_names[row['cluster_kmeans']]}")
    print(f"Suggested archetype: {archetype_names[row['best_match_cluster']]}")
    print(f"Match score:         {row['match_score']:.2f}")
    print(f"Why chose field:     {row[q12]}")
    print(f"Considered changing: {row[q36]}")
    print(f"Choose again?        {row[q39]}")