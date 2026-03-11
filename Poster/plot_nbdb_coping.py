import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Load CSVs ---
df_nb = pd.read_csv("nbHigh.csv")      # Nurturer Agents
df_db = pd.read_csv("dbHigh_lh.csv")   # Defender Agents

# --- Rename 'cry' to 'Prosocial Crying' if it exists ---
if 'cry' in df_nb.columns:
    df_nb = df_nb.rename(columns={'cry':'Prosocial Crying'})
if 'cry' in df_db.columns:
    df_db = df_db.rename(columns={'cry':'Prosocial Crying'})

# --- Metrics to analyze ---
metrics = ['Prosocial Crying', 'learned_helplessness', 'flee']

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
summary = pd.DataFrame(columns=['Agent Type','Metric','Mean','95% CI'])

# --- Compute summary for both agent types ---
for metric in metrics:
    # Nurturer
    if metric in df_nb.columns:
        mean, ci = summarize_column(df_nb[metric].dropna())
        summary.loc[len(summary)] = ['Nurturer', metric, mean, ci]
    else:
        summary.loc[len(summary)] = ['Nurturer', metric, 0, 0]

    # Defender
    if metric in df_db.columns:
        mean, ci = summarize_column(df_db[metric].dropna())
        summary.loc[len(summary)] = ['Defender', metric, mean, ci]
    else:
        summary.loc[len(summary)] = ['Defender', metric, 0, 0]

# --- Print summary table ---
print("\nSummary Table:")
print(summary)

# --- Plotting ---
agent_types = ['Nurturer', 'Defender']
colors = ['skyblue', 'salmon']

fig, axes = plt.subplots(1, len(metrics), figsize=(16,5), sharey=True)

for i, metric in enumerate(metrics):
    means = []
    cis = []
    for agent in agent_types:
        row = summary[(summary['Agent Type']==agent) & (summary['Metric']==metric)]
        means.append(row['Mean'].values[0])
        cis.append(row['95% CI'].values[0])

    bars = axes[i].bar(agent_types, means, yerr=cis, capsize=5, color=colors)
    axes[i].set_title(metric.replace('_',' ').title())
    axes[i].set_ylabel("Mean ± 95% CI")
    axes[i].grid(axis='y', alpha=0.7)

    # Annotate mean ± CI
    for xi, mean, ci in zip(range(len(agent_types)), means, cis):
        axes[i].text(xi, mean + ci + 0.5, f"{mean:.1f} ± {ci:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
