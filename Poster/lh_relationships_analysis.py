import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Load the CSV ---
df = pd.read_csv("dbHigh_lh.csv")

# --- Define groups ---
cutoff = 10
df['group'] = np.where(df['learned_helplessness'] <= cutoff, 'Resilient', 'Vulnerable')  # 'Vulnerable' instead of 'Traumatized'

# --- Metrics function ---
def summarize_column(df_col):
    """
    Returns mean, std, and 95% CI
    """
    n = len(df_col)
    mean = np.mean(df_col)
    std = np.std(df_col, ddof=1)  # sample std
    se = std / np.sqrt(n)
    ci = se * stats.t.ppf(0.975, n-1)  # 95% CI
    return mean, std, ci

# --- Prepare summary table ---
columns_to_analyze = ['aggressiveness_in_relationship', 'healthy_friendliness']
groups = ['All', 'Resilient', 'Vulnerable']
summary = pd.DataFrame(columns=['Group','Variable','Mean','Std','95% CI'])

for var in columns_to_analyze:
    # All agents
    mean, std, ci = summarize_column(df[var].dropna())
    summary = pd.concat([summary, pd.DataFrame({'Group':['All'], 'Variable':[var], 'Mean':[mean], 'Std':[std], '95% CI':[ci]})])
    
    # Resilient
    mean, std, ci = summarize_column(df[df['group']=='Resilient'][var].dropna())
    summary = pd.concat([summary, pd.DataFrame({'Group':['Resilient'], 'Variable':[var], 'Mean':[mean], 'Std':[std], '95% CI':[ci]})])
    
    # Vulnerable
    mean, std, ci = summarize_column(df[df['group']=='Vulnerable'][var].dropna())
    summary = pd.concat([summary, pd.DataFrame({'Group':['Vulnerable'], 'Variable':[var], 'Mean':[mean], 'Std':[std], '95% CI':[ci]})])

print("\nSummary Table:")
print(summary)

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(12,6))
colors = ['skyblue', 'green', 'red']

for i, var in enumerate(columns_to_analyze):
    means = []
    cis = []
    for grp in groups:
        means.append(summary[(summary['Group']==grp) & (summary['Variable']==var)]['Mean'].values[0])
        cis.append(summary[(summary['Group']==grp) & (summary['Variable']==var)]['95% CI'].values[0])
    
    bars = axes[i].bar(groups, means, yerr=cis, capsize=5, color=colors)
    axes[i].set_title(var.replace('_',' ').title())
    axes[i].set_ylabel("Mean ± 95% CI")
    axes[i].grid(axis='y', alpha=0.7)
    
    # --- Add mean ± CI numbers on top of each bar ---
    for xi, mean, ci in zip(range(len(groups)), means, cis):
        axes[i].text(xi, mean + ci + 1, f"{mean:.1f} ± {ci:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
