import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Load CSVs ---
df_high = pd.read_csv("cryfleeHigh_dominance.csv")
df_low  = pd.read_csv("cryfleeLow_dominance.csv")

# --- Rename for poster ---
df_high = df_high.rename(columns={'cry': 'Prosocial Crying'})
df_low  = df_low.rename(columns={'cry': 'Prosocial Crying'})

df_high = df_high.rename(columns={'flee': 'Fleeing'})
df_low  = df_low.rename(columns={'flee': 'Fleeing'})

# --- Metrics to analyze ---
actions = ['Prosocial Crying', 'Fleeing']

# --- Function to compute mean and 95% CI ---
def summarize_column(values):
    n = len(values)
    mean = values.mean()
    if n > 1:
        sem = stats.sem(values)
        ci = sem * stats.t.ppf(0.975, n - 1)
    else:
        ci = 0.0
    return mean, ci

# --- Prepare summary table ---
summary = pd.DataFrame(columns=['ACE Exposure','Group','Action','Mean','95% CI'])

# --- Helper to compute groups ---
def summarize_ace(df, ace_label):
    groups = ['All', 'Cry-dominant', 'Flee-dominant']
    for action in actions:
        # All agents
        mean, ci = summarize_column(df[action].dropna())
        summary.loc[len(summary)] = [ace_label, 'All', action, mean, ci]

        # Cry-dominant
        cry_dom = df[df['dominance_state'] == 'cry_dominant']
        mean, ci = summarize_column(cry_dom[action].dropna())
        summary.loc[len(summary)] = [ace_label, 'Cry-dominant', action, mean, ci]

        # Flee-dominant
        flee_dom = df[df['dominance_state'] == 'flee_dominant']
        mean, ci = summarize_column(flee_dom[action].dropna())
        summary.loc[len(summary)] = [ace_label, 'Flee-dominant', action, mean, ci]

# --- Summarize High and Minimal ACE ---
summarize_ace(df_high, 'High ACE')
summarize_ace(df_low, 'Minimal ACE')

# --- Print summary table ---
print("\nSummary Table:")
print(summary)

# --- Plotting ---
groups_plot = ['All', 'Cry-dominant', 'Flee-dominant']
ace_plot = ['High ACE', 'Minimal ACE']
colors = {'High ACE':'skyblue', 'Minimal ACE':'orange'}

fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

for i, action in enumerate(actions):
    means = []
    cis = []
    labels = []

    for ace in ace_plot:
        for grp in groups_plot:
            row = summary[(summary['ACE Exposure']==ace) & (summary['Group']==grp) & (summary['Action']==action)]
            means.append(row['Mean'].values[0])
            cis.append(row['95% CI'].values[0])
            labels.append(f"{ace}\n{grp}")

    bars = axes[i].bar(labels, means, yerr=cis, capsize=5, color=[colors[label.split('\n')[0]] for label in labels])
    axes[i].set_title(action)
    axes[i].set_ylabel("Mean Occurrences ± 95% CI")
    axes[i].grid(axis='y', alpha=0.7)

    # Annotate mean ± CI
    for xi, mean, ci in zip(range(len(labels)), means, cis):
        axes[i].text(xi, mean + ci + 1, f"{mean:.1f} ± {ci:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
