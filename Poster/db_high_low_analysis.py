import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Load CSV ---
df = pd.read_csv("dbHighLow.csv")

# --- Define metrics to analyze ---
metrics = {
    'learned_helplessness': 'learned_helplessness',
    'aggressiveness_in_relationship': 'aggressiveness_in_relationship',
    'healthy_friendliness': 'healthy_friendliness',
    'misdirected_aggression': 'bully_behavior'  # renamed for poster
}

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
summary = pd.DataFrame(columns=['Group','Variable','Mean','95% CI'])

# --- High ACE: All, Resilient, Vulnerable ---
cutoff_lh = 10
df['high_ACE_status'] = np.where(df['learned_helplessness'] <= cutoff_lh, 'Resilient', 'Vulnerable')

# --- 1) All high ACE agents ---
subset_all = df[['learned_helplessness','aggressiveness_in_relationship','healthy_friendliness','bully_behavior']]
for var, col in metrics.items():
    mean, ci = summarize_column(subset_all[col].dropna())
    summary = pd.concat([summary, pd.DataFrame({'Group':['All'], 'Variable':[var], 'Mean':[mean], '95% CI':[ci]})])

# --- 2) Resilient and Vulnerable high ACE ---
for grp in ['Resilient','Vulnerable']:
    subset = df[df['high_ACE_status'] == grp]
    for var, col in metrics.items():
        mean, ci = summarize_column(subset[col].dropna())
        summary = pd.concat([summary, pd.DataFrame({'Group':[grp], 'Variable':[var], 'Mean':[mean], '95% CI':[ci]})])

# --- 3) Minimal ACE (No ACE) ---
no_ace_cols = {
    'learned_helplessness': 0,  # no learned helplessness
    'aggressiveness_in_relationship': 'low_aggressiveness_in_relationship',
    'healthy_friendliness': 'low_healthy_friendliness',
    'misdirected_aggression': 'low_bully_behavior'
}

for var, col in no_ace_cols.items():
    if col == 0:  # LH = 0 for no ACE
        mean, ci = 0, 0
    else:
        mean, ci = summarize_column(df[col].dropna())
    summary = pd.concat([summary, pd.DataFrame({'Group':['Minimal ACE'], 'Variable':[var], 'Mean':[mean], '95% CI':[ci]})])

# --- Print summary table ---
print("\nSummary Table:")
print(summary)

# --- Plotting ---
groups_to_plot = ['All', 'Resilient', 'Vulnerable', 'Minimal ACE']
colors = ['skyblue', 'green', 'red', 'orange']

fig, axes = plt.subplots(2, 2, figsize=(14,10))
axes = axes.flatten()

for i, var in enumerate(metrics.keys()):
    means = []
    cis = []
    for grp in groups_to_plot:
        means.append(summary[(summary['Group']==grp) & (summary['Variable']==var)]['Mean'].values[0])
        cis.append(summary[(summary['Group']==grp) & (summary['Variable']==var)]['95% CI'].values[0])
    
    bars = axes[i].bar(groups_to_plot, means, yerr=cis, capsize=5, color=colors)
    axes[i].set_title(var.replace('_',' ').title())
    axes[i].set_ylabel("Mean ± 95% CI")
    axes[i].grid(axis='y', alpha=0.7)
    
    # Annotate mean ± 95% CI
    for xi, mean, ci in zip(range(len(groups_to_plot)), means, cis):
        axes[i].text(xi, mean + ci + 1, f"{mean:.1f} ± {ci:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
