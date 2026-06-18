import pandas as pd

df = pd.read_csv("AcaDNA_final.csv")
df = df[df["data_source"] == "real"].reset_index(drop=True)

# ── Archetype (Module 1) ──────────────────────────────
archetype_names = {
    0: "The Passionate Pretender",
    1: "The Pressure Procrastinator",
    2: "The Serene Organizer",
    3: "The Exhausted Masker",
    4: "The Night Owl"
}
df["archetype"] = df["cluster_kmeans"].map(archetype_names)

# ── Mask level (Module 2) ─────────────────────────────
mask_map = {
    "No, I am always myself / لا، أنا نفسي دائماً": "No_mask",
    "A little / قليلاً": "No_mask",
    "Sometimes yes / أحياناً نعم": "Mask",
    "Often / غالباً": "Mask",
    "Almost all the time / تقريباً طوال الوقت": "Mask"
}
df["mask_level"] = df[df.columns[19]].map(mask_map)

# ── Procrastination type (Module 2) ───────────────────
proc_map = {
    "I plan well in advance / أخطط مسبقاً جيداً": "Disciplined",
    "Pressure motivates me / الضغط يحفزني": "Optimizer",
    "I meet them but at the last moment / ألتزم بها لكن في اللحظة الأخيرة": "Avoider",
    "I sometimes miss them / أتجاوزها أحياناً": "Overwhelmed",
    "I often ignore them / أتجاهلها غالباً": "Overwhelmed"
}
df["procrastination_type"] = df[df.columns[28]].map(proc_map)

# ── Check the unified profile ─────────────────────────
profile_cols = ["archetype", "mask_level", "procrastination_type", "match_score", "mal_oriente"]
print(df[profile_cols].head(10))
print(df[profile_cols].isnull().sum())










import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

q40 = df.columns[40]
drop_cols = ["Horodateur", "data_source", q40, "cluster_kmeans", "cluster_dbscan",
             "pca1", "pca2", "mal_oriente", "best_match_cluster", "match_score",
             "archetype", "mask_level", "procrastination_type"]
X = df.drop(columns=drop_cols)

for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sim_matrix = cosine_similarity(X_scaled)

def get_similar_tips(student_idx, k=5):
    sims = sim_matrix[student_idx].copy()
    sims[student_idx] = -1  # exclude self
    top_k_idx = np.argsort(sims)[-k:][::-1]
    return df.loc[top_k_idx, q40].tolist()

# Test it on a couple of students
print("Student 2 (Exhausted Masker) similar tips:")
print(get_similar_tips(2, k=5))
print("\nStudent 3 (Serene Organizer) similar tips:")
print(get_similar_tips(3, k=5))






from mlxtend.frequent_patterns import apriori, association_rules

mining_cols = ["archetype", "mask_level", "procrastination_type",
               df.columns[9], df.columns[21], df.columns[26], df.columns[34]]
mining_df = df[mining_cols].astype(str)

transactions = pd.get_dummies(mining_df)

frequent_itemsets = apriori(transactions, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules.sort_values("lift", ascending=False)

for idx, row in rules.head(15).iterrows():
    antecedents = ", ".join(list(row["antecedents"]))
    consequents = ", ".join(list(row["consequents"]))
    print(f"IF [{antecedents}] THEN [{consequents}]")
    print(f"   support={row['support']:.2f} confidence={row['confidence']:.2f} lift={row['lift']:.2f}")
    print()




def get_matching_insights(student_row, rules, max_insights=2):
    student_traits = {
        f"archetype_{student_row['archetype']}",
        f"mask_level_{student_row['mask_level']}",
        f"procrastination_type_{student_row['procrastination_type']}"
    }
    matches = []
    seen = set()
    for _, rule in rules.iterrows():
        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])
        if not antecedents.issubset(student_traits):
            continue
        # skip circular insights that just restate a known trait
        if any(c.startswith(("archetype_", "mask_level_", "procrastination_type_")) for c in consequents):
            continue
        consequent_str = ", ".join(consequents)
        if consequent_str in seen:
            continue
        seen.add(consequent_str)
        matches.append((consequent_str, rule["confidence"], rule["lift"]))
    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    return matches[:max_insights]

def generate_success_plan(student_idx):
    row = df.iloc[student_idx]
    insights = get_matching_insights(row, rules)
    tips = get_similar_tips(student_idx, k=5)

    plan = []
    for consequent, conf, lift in insights:
        plan.append(f"Pattern detected: students like you ({row['archetype']}, {row['mask_level']}, {row['procrastination_type']}) often also: {consequent} (confidence={conf:.0%})")

    unique_tips = [t.strip() for t in tips if t and t.strip() and t.strip().lower() != "n/a"]
    plan.extend(unique_tips[:max(0, 3 - len(plan))])
    return plan[:3]

print("Plan for student 2:")
for tip in generate_success_plan(2):
    print(" -", tip)

print("\nPlan for student 3:")
for tip in generate_success_plan(3):
    print(" -", tip)