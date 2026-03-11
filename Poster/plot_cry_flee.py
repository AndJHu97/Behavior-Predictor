import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Load data ---
df = pd.read_csv("cryflee_dominance.csv")

# --- Define columns and groups ---
columns_to_analyze = ['cry', 'flee']
groups = ['All', 'Cry-dominant', 'Flee-dominant']

# --- Function to calculate mean and 95% CI ---
def summarize_column(values):
    n = len(values)
    mean = values.mean()
    if n > 1:
        sem = stats.sem(values)
        ci = sem * stats.t.ppf(0.975, df=n - 1)
    else:
        ci = 0.0
    return mean, ci

# --- Prepare summary table ---
summary = pd.DataFrame(columns=['Group','Variable','Mean','95% CI'])

for var in columns_to_analyze:
    # All agents
    mean, ci = summarize_column(df[var].dropna())
    summary = pd.concat([summary, pd.DataFrame({'Group':['All'], 'Variable':[var], 'Mean':[mean], '95% CI':[ci]})])
    
    # Cry-dominant
    mean, ci = summarize_column(df.loc[df['dominance_state']=='cry_dominant', var].dropna())
    summary = pd.concat([summary, pd.DataFrame({'Group':['Cry-dominant'], 'Variable':[var], 'Mean':[mean], '95% CI':[ci]})])
    
    # Flee-dominant
    mean, ci = summarize_column(df.loc[df['dominance_state']=='flee_dominant', var].dropna())
    summary = pd.concat([summary, pd.DataFrame({'Group':['Flee-dominant'], 'Variable':[var], 'Mean':[mean], '95% CI':[ci]})])

print("\nSummary Table:")
print(summary)

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(12,5))

colors = ['skyblue', 'green', 'red']  # All, Cry-dominant, Flee-dominant

for i, var in enumerate(columns_to_analyze):
    means = []
    cis = []
    for grp in groups:
        means.append(summary[(summary['Group']==grp) & (summary['Variable']==var)]['Mean'].values[0])
        cis.append(summary[(summary['Group']==grp) & (summary['Variable']==var)]['95% CI'].values[0])
    
    axes[i].bar(groups, means, yerr=cis, capsize=5, color=colors)
    
    # Annotate mean ± 95% CI
    for xi, mean, ci in zip(range(len(groups)), means, cis):
        axes[i].text(xi, mean + ci + 1, f"{mean:.1f} ± {ci:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[i].set_title(var.title())
    axes[i].set_ylabel("Occurrences per run")
    axes[i].grid(axis='y', alpha=0.7)

plt.tight_layout()
plt.show()
