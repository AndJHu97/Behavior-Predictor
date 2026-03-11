import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

df = pd.read_csv("cryfleeLow_dominance.csv")

def analyze_behavior(df, behavior):
    """
    behavior: 'cry' or 'flee'
    """
    groups = {
        "All": df[behavior].dropna(),
        "Cry-dominant": df.loc[df["dominance_state"] == "cry_dominant", behavior].dropna(),
        "Flee-dominant": df.loc[df["dominance_state"] == "flee_dominant", behavior].dropna(),
    }

    print(f"\n=== {behavior.upper()} ANALYSIS ===")

    # --- Kruskal–Wallis ---
    kw_stat, kw_p = stats.kruskal(*groups.values())
    print(f"Kruskal–Wallis H = {kw_stat:.2f}, p = {kw_p:.3e}")

    # --- One-way ANOVA ---
    f_stat, f_p = stats.f_oneway(*groups.values())
    print(f"ANOVA F = {f_stat:.2f}, p = {f_p:.3e}")

    # --- Pairwise comparisons (Mann–Whitney U) ---
    rows = []
    for (name1, data1), (name2, data2) in combinations(groups.items(), 2):
        u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        rows.append({
            "Behavior": behavior,
            "Comparison": f"{name1} vs {name2}",
            "U": u_stat,
            "p_value": p_val
        })

    return pd.DataFrame(rows)

# Run analyses
flee_results = analyze_behavior(df, "flee")
cry_results = analyze_behavior(df, "cry")

# Combine for display
results_table = pd.concat([flee_results, cry_results], ignore_index=True)
print("\n=== Pairwise Post-hoc Tests ===")
print(results_table)
