#!/usr/bin/env python3
"""
Figure 4B: Adaptive Behavior Emergence Analysis
Analyzes how adaptive behaviors emerge across adversity conditions (No ACE, Moderate, ACE)
Using Kruskal-Wallis + Mann-Whitney U statistical tests with Holm correction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from glob import glob
import warnings
import argparse
import os

warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(invalid='ignore')

# Adaptive behavior variable mappings from simulation.py
ADAPTIVE_BEHAVIORS = {
    'protective_behavior': 'Protective Behavior',
    'healthy_friendliness': 'Healthy Friendliness',
    'willingness_to_flee': 'Adaptive Avoidance',
    'community_trusting_vulnerability': 'Help-Seeking Vulnerability',
    'hopefulness': 'Positive Expectancy',
}

MALADAPTIVE_BEHAVIOR_COLUMNS = [
    'fearful_withdrawn_relationship',
    'dangerous_trust',
    'bully_behavior',
    'aggressive_withdrawn_relationship',
    'learned_helplessness',
    'cynical',
]

TOTAL_BEHAVIOR_COLUMNS = list(dict.fromkeys(list(ADAPTIVE_BEHAVIORS.keys()) + MALADAPTIVE_BEHAVIOR_COLUMNS))
EXPECTANCY_COLUMNS = ['hopefulness', 'cynical']
LEARNED_HELPLESSNESS_COLUMN = 'learned_helplessness'
EXPECTANCY_DENOM_COLUMNS = ['hopefulness', 'cynical', LEARNED_HELPLESSNESS_COLUMN]
COMPLEX_ACTION_COLUMNS = [c for c in TOTAL_BEHAVIOR_COLUMNS if c not in EXPECTANCY_COLUMNS]

def parse_condition(filename):
    """Extract adversity condition from filename"""
    name = filename.lower()
    if 'highthreat' in name:
        return 'ACE'
    elif 'moderatethreat' in name:
        return 'Moderate'
    elif 'default' in name or 'lowthreat' in name:
        return 'No ACE'
    return None

def extract_adaptive_behaviors(csv_files, role):
    """
    Extract adaptive behavior data from CSV files
    Returns DataFrame with run-level adaptive behavior percentages
    """
    all_data = []
    
    for csv_file in csv_files:
        condition = parse_condition(csv_file)
        if condition is None:
            print(f"Skipping {csv_file}: could not determine condition")
            continue
        
        try:
            # Read CSV with parameters at top
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Find where the run data starts (header line with "Run")
            header_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('Run,'):
                    header_idx = i
                    break
            
            if header_idx == 0:
                print(f"Could not find header in {csv_file}")
                continue
            
            # Read only the run data
            df_runs = pd.read_csv(csv_file, skiprows=header_idx)
            
            if df_runs.empty:
                print(f"No run data in {csv_file}")
                continue
            
            # Convert to numeric (already should be from CSV)
            for col in df_runs.columns:
                if col not in ['Run', 'source_file']:
                    try:
                        df_runs[col] = pd.to_numeric(df_runs[col], errors='coerce')
                    except:
                        pass
            
            # Calculate denominators: complex behaviors and expectancies.
            if all(col in df_runs.columns for col in TOTAL_BEHAVIOR_COLUMNS):
                df_runs['total_complex_behaviors'] = df_runs[COMPLEX_ACTION_COLUMNS].sum(axis=1)
                df_runs['total_expectancies'] = df_runs[EXPECTANCY_DENOM_COLUMNS].sum(axis=1)
            else:
                print(f"Missing behavior columns in {csv_file}")
                continue
            
            # Process each adaptive behavior
            for behavior_var, behavior_name in ADAPTIVE_BEHAVIORS.items():
                if behavior_var not in df_runs.columns:
                    print(f"Warning: {behavior_var} not found in {csv_file}")
                    continue
                
                if behavior_var in EXPECTANCY_COLUMNS:
                    denom_col = 'total_expectancies'
                else:
                    denom_col = 'total_complex_behaviors'

                df_runs[f'{behavior_var}_denominator'] = df_runs[denom_col]
                # Calculate percentage using behavior-appropriate denominator.
                df_runs[f'{behavior_var}_pct'] = (df_runs[behavior_var] / df_runs[denom_col].replace(0, np.nan)) * 100
                df_runs[f'{behavior_var}_pct'] = df_runs[f'{behavior_var}_pct'].fillna(0)
            
            # Add metadata
            df_runs['source_file'] = os.path.basename(csv_file)
            df_runs['condition'] = condition
            df_runs['role'] = role
            
            all_data.append(df_runs)
            print(f"Extracted {len(df_runs)} runs from {csv_file} ({condition}, {role})")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

def holms_correction(p_values):
    """Apply Holm step-down correction to p-values"""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    corrected = sorted_p * np.arange(n, 0, -1)
    corrected = np.minimum(corrected, 1.0)
    
    # Unsort back to original order
    result = np.empty_like(corrected)
    result[sorted_indices] = corrected
    return result

def perform_statistical_tests(df_combined):
    """
    Perform Kruskal-Wallis and Mann-Whitney U tests for adaptive behaviors
    """
    results = []
    
    for behavior_var, behavior_name in ADAPTIVE_BEHAVIORS.items():
        pct_col = f'{behavior_var}_pct'
        
        # Check if column exists in data
        if pct_col not in df_combined.columns:
            print(f"Skipping {behavior_var}: column not found")
            continue
        
        # Get data by condition
        no_ace_data = df_combined[df_combined['condition'] == 'No ACE'][pct_col].dropna().values
        moderate_data = df_combined[df_combined['condition'] == 'Moderate'][pct_col].dropna().values
        ace_data = df_combined[df_combined['condition'] == 'ACE'][pct_col].dropna().values
        
        # Kruskal-Wallis test (across all 3 conditions)
        h_stat, p_kw = kruskal(no_ace_data, moderate_data, ace_data)
        
        # Compute medians and IQRs for all conditions
        no_ace_median = np.median(no_ace_data) if len(no_ace_data) > 0 else 0
        no_ace_q1 = np.percentile(no_ace_data, 25) if len(no_ace_data) > 0 else 0
        no_ace_q3 = np.percentile(no_ace_data, 75) if len(no_ace_data) > 0 else 0
        no_ace_n = len(no_ace_data)
        
        moderate_median = np.median(moderate_data) if len(moderate_data) > 0 else 0
        moderate_q1 = np.percentile(moderate_data, 25) if len(moderate_data) > 0 else 0
        moderate_q3 = np.percentile(moderate_data, 75) if len(moderate_data) > 0 else 0
        moderate_n = len(moderate_data)
        
        ace_median = np.median(ace_data) if len(ace_data) > 0 else 0
        ace_q1 = np.percentile(ace_data, 25) if len(ace_data) > 0 else 0
        ace_q3 = np.percentile(ace_data, 75) if len(ace_data) > 0 else 0
        ace_n = len(ace_data)
        
        # Mann-Whitney U post-hoc tests: (No ACE vs ACE) and (Moderate vs ACE)
        u_no_ace_ace, p_no_ace_ace = mannwhitneyu(no_ace_data, ace_data, alternative='two-sided') if len(no_ace_data) > 0 and len(ace_data) > 0 else (np.nan, np.nan)
        u_mod_ace, p_mod_ace = mannwhitneyu(moderate_data, ace_data, alternative='two-sided') if len(moderate_data) > 0 and len(ace_data) > 0 else (np.nan, np.nan)
        
        # Holm correction for 2 post-hoc tests
        p_raw = np.array([p_no_ace_ace, p_mod_ace])
        p_holm = holms_correction(p_raw)
        p_no_ace_ace_holm = p_holm[0]
        p_mod_ace_holm = p_holm[1]
        
        results.append({
            'Behavior': behavior_name,
            'H_stat': h_stat,
            'p_kruskal_wallis': p_kw,
            'Significant_KW': 'Yes' if p_kw < 0.05 else 'No',
            'No_ACE_Median': no_ace_median,
            'No_ACE_Q1': no_ace_q1,
            'No_ACE_Q3': no_ace_q3,
            'No_ACE_N': no_ace_n,
            'Moderate_Median': moderate_median,
            'Moderate_Q1': moderate_q1,
            'Moderate_Q3': moderate_q3,
            'Moderate_N': moderate_n,
            'ACE_Median': ace_median,
            'ACE_Q1': ace_q1,
            'ACE_Q3': ace_q3,
            'ACE_N': ace_n,
            'U_NoACE_vs_ACE': u_no_ace_ace,
            'p_NoACE_vs_ACE_raw': p_no_ace_ace,
            'p_NoACE_vs_ACE_holm': p_no_ace_ace_holm,
            'NoACE_vs_ACE_Sig': 'Yes' if p_no_ace_ace_holm < 0.05 else 'No',
            'U_Moderate_vs_ACE': u_mod_ace,
            'p_Moderate_vs_ACE_raw': p_mod_ace,
            'p_Moderate_vs_ACE_holm': p_mod_ace_holm,
            'Moderate_vs_ACE_Sig': 'Yes' if p_mod_ace_holm < 0.05 else 'No',
        })
    
    return pd.DataFrame(results)

def create_visualization(df_combined, output_dir):
    """Create grouped bar plot with random gray baseline bars."""
    os.makedirs(output_dir, exist_ok=True)
    
    behaviors = list(ADAPTIVE_BEHAVIORS.values())
    conditions = ['No ACE', 'Moderate', 'ACE']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    rng = np.random.default_rng(42)
    role_observed = ['NB', 'DB']
    role_random = ['Random NB', 'Random DB']

    for idx, behavior_name in enumerate(behaviors):
        ax = axes[idx]
        behavior_var = [k for k, v in ADAPTIVE_BEHAVIORS.items() if v == behavior_name][0]
        pct_col = f'{behavior_var}_pct'

        colors = ['#1f77b4', '#ff7f0e', '#d62728']
        x_pos = np.arange(len(conditions))

        width = 0.34
        for i, condition in enumerate(conditions):
            observed_vals = df_combined[
                (df_combined['condition'] == condition) & (df_combined['role'].isin(role_observed))
            ][pct_col].dropna().values
            random_vals = df_combined[
                (df_combined['condition'] == condition) & (df_combined['role'].isin(role_random))
            ][pct_col].dropna().values

            if observed_vals.size > 0:
                obs_med = np.median(observed_vals)
                obs_q1 = np.percentile(observed_vals, 25)
                obs_q3 = np.percentile(observed_vals, 75)
                obs_x = x_pos[i] - (width / 2)
                ax.bar(obs_x, obs_med, width=width, color=colors[i], alpha=0.75, edgecolor='black', linewidth=1.2)
                ax.errorbar(obs_x, obs_med, yerr=[[obs_med - obs_q1], [obs_q3 - obs_med]], fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)
                obs_jitter = rng.normal(0, width * 0.12, size=observed_vals.size)
                ax.scatter(obs_x + obs_jitter, observed_vals, alpha=0.35, s=18, color=colors[i], edgecolors='black', linewidth=0.4)

            if random_vals.size > 0:
                rnd_med = np.median(random_vals)
                rnd_q1 = np.percentile(random_vals, 25)
                rnd_q3 = np.percentile(random_vals, 75)
                rnd_x = x_pos[i] + (width / 2)
                ax.bar(rnd_x, rnd_med, width=width, color='#cfcfcf', alpha=0.9, edgecolor='black', linewidth=1.2)
                ax.errorbar(rnd_x, rnd_med, yerr=[[rnd_med - rnd_q1], [rnd_q3 - rnd_med]], fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)
                rnd_jitter = rng.normal(0, width * 0.12, size=random_vals.size)
                ax.scatter(rnd_x + rnd_jitter, random_vals, alpha=0.35, s=18, color='#9a9a9a', edgecolors='black', linewidth=0.4)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions, fontsize=11)
        if behavior_var in EXPECTANCY_COLUMNS:
            ax.set_ylabel('% of Total Expectancies', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('% of Total Complex Behaviors', fontsize=11, fontweight='bold')
        ax.set_title(behavior_name, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)

        if idx == 0:
            from matplotlib.patches import Patch
            legend_items = [
                Patch(facecolor='#1f77b4', edgecolor='black', label='Observed (NB+DB)'),
                Patch(facecolor='#cfcfcf', edgecolor='black', label='Random baseline'),
            ]
            ax.legend(handles=legend_items, fontsize=10, loc='upper left')
    
    # Remove the 6th subplot (we only have 5 adaptive behaviors)
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure4b_adaptive_emergence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()

def print_summary_table(stats_df):
    """Print formatted summary table"""
    print("\n" + "="*120)
    print("ADAPTIVE BEHAVIOR EMERGENCE ANALYSIS - Summary Statistics")
    print("="*120)
    
    for _, row in stats_df.iterrows():
        print(f"\n{row['Behavior']}:")
        print(f"  Kruskal-Wallis: H={row['H_stat']:.2f}, p={row['p_kruskal_wallis']:.6f} ({row['Significant_KW']})")
        print(f"  No ACE:       Median={row['No_ACE_Median']:.2f}% [{row['No_ACE_Q1']:.2f}, {row['No_ACE_Q3']:.2f}] (n={int(row['No_ACE_N'])})")
        print(f"  Moderate:     Median={row['Moderate_Median']:.2f}% [{row['Moderate_Q1']:.2f}, {row['Moderate_Q3']:.2f}] (n={int(row['Moderate_N'])})")
        print(f"  ACE:          Median={row['ACE_Median']:.2f}% [{row['ACE_Q1']:.2f}, {row['ACE_Q3']:.2f}] (n={int(row['ACE_N'])})")
        print(f"  Post-hoc No ACE vs ACE: U={row['U_NoACE_vs_ACE']:.1f}, p_raw={row['p_NoACE_vs_ACE_raw']:.6f}, p_holm={row['p_NoACE_vs_ACE_holm']:.6f} ({row['NoACE_vs_ACE_Sig']})")
        print(f"  Post-hoc Moderate vs ACE: U={row['U_Moderate_vs_ACE']:.1f}, p_raw={row['p_Moderate_vs_ACE_raw']:.6f}, p_holm={row['p_Moderate_vs_ACE_holm']:.6f} ({row['Moderate_vs_ACE_Sig']})")

def main():
    parser = argparse.ArgumentParser(description='Figure 4B: Adaptive Behavior Emergence Analysis')
    parser.add_argument('--NB', type=str, default='multiple_runs_NB_*Threat*.csv', help='NB file pattern')
    parser.add_argument('--DB', type=str, default='multiple_runs_DB_*Threat*.csv', help='DB file pattern')
    parser.add_argument('--RandomNB', type=str, default='multiple_runs_random_NB_*Threat*.csv', help='Random NB file pattern')
    parser.add_argument('--RandomDB', type=str, default='multiple_runs_random_DB_*Threat*.csv', help='Random DB file pattern')
    parser.add_argument('--output', type=str, default=os.path.join('FinalProject', 'figure4'), help='Output directory')
    
    args = parser.parse_args()
    
    # Find files
    nb_files = sorted(glob(args.NB))
    db_files = sorted(glob(args.DB))
    random_nb_files = sorted(glob(args.RandomNB))
    random_db_files = sorted(glob(args.RandomDB))
    
    print(f"Found {len(nb_files)} NB, {len(db_files)} DB, {len(random_nb_files)} Random NB, {len(random_db_files)} Random DB files")
    
    if not (nb_files or db_files or random_nb_files or random_db_files):
        print("Error: No CSV files found")
        return
    
    # Extract data
    df_nb = extract_adaptive_behaviors(nb_files, 'NB')
    df_db = extract_adaptive_behaviors(db_files, 'DB')
    df_random_nb = extract_adaptive_behaviors(random_nb_files, 'Random NB')
    df_random_db = extract_adaptive_behaviors(random_db_files, 'Random DB')
    
    # Combine
    df_combined = pd.concat([df_nb, df_db, df_random_nb, df_random_db], ignore_index=True)
    
    if df_combined.empty:
        print("Error: No data extracted")
        return
    
    print(f"\nTotal runs extracted: {len(df_combined)}")
    print(f"Conditions: {df_combined['condition'].unique()}")
    print(f"Roles: {df_combined['role'].unique()}")
    
    # Perform statistical tests
    stats_df = perform_statistical_tests(df_combined[df_combined['role'].isin(['NB', 'DB'])])
    
    # Print summary
    print_summary_table(stats_df)
    
    # Create visualization
    create_visualization(df_combined, args.output)
    
    # Save raw data
    os.makedirs(args.output, exist_ok=True)
    
    # Export raw percentages
    raw_export = []
    for behavior_var, behavior_name in ADAPTIVE_BEHAVIORS.items():
        pct_col = f'{behavior_var}_pct'
        for idx, row in df_combined.iterrows():
            raw_export.append({
                'source_file': row['source_file'],
                'condition': row['condition'],
                'role': row['role'],
                'run': row.get('Run', idx),
                'behavior': behavior_name,
                'behavior_count': row[behavior_var],
                'total_actions': row[f'{behavior_var}_denominator'],
                'percent_of_total_actions': row[pct_col]
            })
    
    raw_df = pd.DataFrame(raw_export)
    raw_path = os.path.join(args.output, 'figure4b_raw_behavior_percentages.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f"\nExported raw data to {raw_path}")
    
    # Export statistics
    stats_path = os.path.join(args.output, 'figure4b_stats_kruskal.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Exported statistics to {stats_path}")

if __name__ == '__main__':
    main()
