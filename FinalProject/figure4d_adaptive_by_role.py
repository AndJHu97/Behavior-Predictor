#!/usr/bin/env python3
"""
Figure 4D: Adaptive Behavior Emergence Analysis - Split by Role (NB vs DB vs Random)
Analyzes adaptive behaviors across adversity conditions by role
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
    if 'HighThreat' in filename or 'highThreat' in filename:
        return 'ACE'
    elif 'ModerateThreat' in filename:
        return 'Moderate'
    elif 'default' in filename or 'LowThreat' in filename or 'lowThreat' in filename:
        return 'No ACE'
    return None

def parse_role(filename):
    """Extract role (NB, DB, or Random) from filename"""
    if 'random' in filename.lower():
        if 'NB' in filename:
            return 'Random NB'
        elif 'DB' in filename:
            return 'Random DB'
        return 'Random'
    elif 'NB' in filename:
        return 'NB'
    elif 'DB' in filename:
        return 'DB'
    return None

def extract_adaptive_behaviors(csv_files):
    """
    Extract adaptive behavior data from CSV files (NB, DB, and Random)
    Returns DataFrame with run-level adaptive behavior percentages
    """
    all_data = []
    
    for csv_file in csv_files:
        condition = parse_condition(csv_file)
        role = parse_role(csv_file)
        
        if condition is None or role is None:
            print(f"Skipping {csv_file}: could not determine condition or role")
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
            
            # Convert to numeric
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
    
    result = np.empty_like(corrected)
    result[sorted_indices] = corrected
    return result

def perform_statistical_tests(df_combined):
    """
    Perform Kruskal-Wallis and Mann-Whitney U tests for adaptive behaviors by role.
    Tests role differences (NB vs DB) and matched baselines (NB vs Random NB, DB vs Random DB)
    within each condition.
    """
    results = []
    
    for behavior_var, behavior_name in ADAPTIVE_BEHAVIORS.items():
        pct_col = f'{behavior_var}_pct'
        
        if pct_col not in df_combined.columns:
            print(f"Skipping {behavior_var}: column not found")
            continue
        
        # Test within each condition: NB vs DB vs Random
        for condition in ['No ACE', 'Moderate', 'ACE']:
            # Get data by role within this condition
            nb_data = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'NB')][pct_col].dropna().values
            db_data = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'DB')][pct_col].dropna().values
            random_nb_data = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'Random NB')][pct_col].dropna().values
            random_db_data = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'Random DB')][pct_col].dropna().values
            
            # Skip if insufficient data
            if len(nb_data) == 0 or len(db_data) == 0 or len(random_nb_data) == 0 or len(random_db_data) == 0:
                continue
            
            # Kruskal-Wallis test across all four role streams.
            h_stat, p_kw = kruskal(nb_data, db_data, random_nb_data, random_db_data)
            
            # Compute medians and IQRs
            nb_median = np.median(nb_data)
            nb_q1 = np.percentile(nb_data, 25)
            nb_q3 = np.percentile(nb_data, 75)
            nb_n = len(nb_data)
            
            db_median = np.median(db_data)
            db_q1 = np.percentile(db_data, 25)
            db_q3 = np.percentile(db_data, 75)
            db_n = len(db_data)
            
            random_nb_median = np.median(random_nb_data)
            random_nb_q1 = np.percentile(random_nb_data, 25)
            random_nb_q3 = np.percentile(random_nb_data, 75)
            random_nb_n = len(random_nb_data)

            random_db_median = np.median(random_db_data)
            random_db_q1 = np.percentile(random_db_data, 25)
            random_db_q3 = np.percentile(random_db_data, 75)
            random_db_n = len(random_db_data)
            
            # Mann-Whitney U post-hoc tests: NB vs DB, NB vs Random NB, DB vs Random DB
            u_nb_db, p_nb_db = mannwhitneyu(nb_data, db_data, alternative='two-sided')
            u_nb_random, p_nb_random = mannwhitneyu(nb_data, random_nb_data, alternative='two-sided')
            u_db_random, p_db_random = mannwhitneyu(db_data, random_db_data, alternative='two-sided')
            
            # Holm correction for 3 post-hoc tests
            p_raw = np.array([p_nb_db, p_nb_random, p_db_random])
            p_holm = holms_correction(p_raw)
            
            results.append({
                'Behavior': behavior_name,
                'Condition': condition,
                'H_stat': h_stat,
                'p_kruskal_wallis': p_kw,
                'Significant_KW': 'Yes' if p_kw < 0.05 else 'No',
                'NB_Median': nb_median,
                'NB_Q1': nb_q1,
                'NB_Q3': nb_q3,
                'NB_N': nb_n,
                'DB_Median': db_median,
                'DB_Q1': db_q1,
                'DB_Q3': db_q3,
                'DB_N': db_n,
                'Random_NB_Median': random_nb_median,
                'Random_NB_Q1': random_nb_q1,
                'Random_NB_Q3': random_nb_q3,
                'Random_NB_N': random_nb_n,
                'Random_DB_Median': random_db_median,
                'Random_DB_Q1': random_db_q1,
                'Random_DB_Q3': random_db_q3,
                'Random_DB_N': random_db_n,
                'U_NB_vs_DB': u_nb_db,
                'p_NB_vs_DB_raw': p_nb_db,
                'p_NB_vs_DB_holm': p_holm[0],
                'NB_vs_DB_Sig': 'Yes' if p_holm[0] < 0.05 else 'No',
                'U_NB_vs_Random_NB': u_nb_random,
                'p_NB_vs_Random_NB_raw': p_nb_random,
                'p_NB_vs_Random_NB_holm': p_holm[1],
                'NB_vs_Random_NB_Sig': 'Yes' if p_holm[1] < 0.05 else 'No',
                'U_DB_vs_Random_DB': u_db_random,
                'p_DB_vs_Random_DB_raw': p_db_random,
                'p_DB_vs_Random_DB_holm': p_holm[2],
                'DB_vs_Random_DB_Sig': 'Yes' if p_holm[2] < 0.05 else 'No',
            })
    
    return pd.DataFrame(results)

def create_visualization(df_combined, output_dir):
    """Create Figure 4D with 4B-style layout and random dashed baselines."""
    os.makedirs(output_dir, exist_ok=True)
    
    behaviors = list(ADAPTIVE_BEHAVIORS.values())
    conditions = ['No ACE', 'Moderate', 'ACE']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    rng = np.random.default_rng(42)
    cond_colors = {'No ACE': '#1f77b4', 'Moderate': '#ff7f0e', 'ACE': '#d62728'}
    role_offsets = {'NB': -0.18, 'DB': 0.18}
    bar_width = 0.32
    
    for idx, behavior_name in enumerate(behaviors):
        ax = axes[idx]
        behavior_var = [k for k, v in ADAPTIVE_BEHAVIORS.items() if v == behavior_name][0]
        pct_col = f'{behavior_var}_pct'
        
        x_pos = np.arange(len(conditions), dtype=float)

        for i, condition in enumerate(conditions):
            nb_vals = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'NB')][pct_col].dropna().values
            db_vals = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'DB')][pct_col].dropna().values
            random_nb_vals = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'Random NB')][pct_col].dropna().values
            random_db_vals = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'Random DB')][pct_col].dropna().values

            if nb_vals.size > 0:
                nb_med = np.median(nb_vals)
                nb_q1 = np.percentile(nb_vals, 25)
                nb_q3 = np.percentile(nb_vals, 75)
                nb_x = x_pos[i] + role_offsets['NB']
                ax.bar(nb_x, nb_med, bar_width, color=cond_colors[condition], alpha=0.75, edgecolor='black', linewidth=1.2)
                ax.errorbar(nb_x, nb_med, yerr=[[nb_med - nb_q1], [nb_q3 - nb_med]], fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)
                nb_jitter = rng.normal(0, bar_width * 0.12, size=nb_vals.size)
                ax.scatter(nb_x + nb_jitter, nb_vals, alpha=0.35, s=18, color=cond_colors[condition], edgecolors='black', linewidth=0.4)

            if db_vals.size > 0:
                db_med = np.median(db_vals)
                db_q1 = np.percentile(db_vals, 25)
                db_q3 = np.percentile(db_vals, 75)
                db_x = x_pos[i] + role_offsets['DB']
                ax.bar(db_x, db_med, bar_width, color=cond_colors[condition], alpha=0.75, edgecolor='black', linewidth=1.2, hatch='//')
                ax.errorbar(db_x, db_med, yerr=[[db_med - db_q1], [db_q3 - db_med]], fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)
                db_jitter = rng.normal(0, bar_width * 0.12, size=db_vals.size)
                ax.scatter(db_x + db_jitter, db_vals, alpha=0.35, s=18, color=cond_colors[condition], edgecolors='black', linewidth=0.4)

            # Role-matched random baselines as gray dashed guides.
            if random_nb_vals.size > 0:
                y_nb_base = float(np.median(random_nb_vals))
                x_left = x_pos[i] + role_offsets['NB'] - (bar_width * 0.46)
                x_right = x_pos[i] + role_offsets['NB'] + (bar_width * 0.46)
                ax.hlines(y_nb_base, x_left, x_right, colors='#7f7f7f', linestyles='--', linewidth=2.0)

            if random_db_vals.size > 0:
                y_db_base = float(np.median(random_db_vals))
                x_left = x_pos[i] + role_offsets['DB'] - (bar_width * 0.46)
                x_right = x_pos[i] + role_offsets['DB'] + (bar_width * 0.46)
                ax.hlines(y_db_base, x_left, x_right, colors='#7f7f7f', linestyles='-.', linewidth=2.0)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions, fontsize=11)
        if behavior_var in EXPECTANCY_COLUMNS:
            ax.set_ylabel('% of Total Expectancies', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('% of Total Complex Behaviors', fontsize=11, fontweight='bold')
        ax.set_title(behavior_name, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Add legend (only on first subplot)
        if idx == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#1f77b4', edgecolor='black', label='NB (solid)'),
                Patch(facecolor='#1f77b4', edgecolor='black', hatch='//', label='DB (hatched)'),
                Line2D([0], [0], color='#7f7f7f', linestyle='--', linewidth=2, label='Random NB baseline'),
                Line2D([0], [0], color='#7f7f7f', linestyle='-.', linewidth=2, label='Random DB baseline'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Remove 6th subplot (we only have 5 adaptive behaviors)
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure4d_adaptive_by_role.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()

def print_summary_table(stats_df):
    """Print formatted summary table"""
    print("\n" + "="*160)
    print("ADAPTIVE BEHAVIOR BY ROLE ANALYSIS - Summary Statistics")
    print("="*160)
    
    for behavior in stats_df['Behavior'].unique():
        print(f"\n{behavior}:")
        behavior_data = stats_df[stats_df['Behavior'] == behavior]
        
        for condition in ['No ACE', 'Moderate', 'ACE']:
            cond_data = behavior_data[behavior_data['Condition'] == condition]
            if cond_data.empty:
                continue
            
            row = cond_data.iloc[0]
            print(f"\n  {condition}:")
            print(f"    Kruskal-Wallis: H={row['H_stat']:.2f}, p={row['p_kruskal_wallis']:.6f} ({row['Significant_KW']})")
            print(f"    NB:     Median={row['NB_Median']:.2f}% [{row['NB_Q1']:.2f}, {row['NB_Q3']:.2f}] (n={int(row['NB_N'])})")
            print(f"    DB:     Median={row['DB_Median']:.2f}% [{row['DB_Q1']:.2f}, {row['DB_Q3']:.2f}] (n={int(row['DB_N'])})")
            print(f"    Random NB: Median={row['Random_NB_Median']:.2f}% [{row['Random_NB_Q1']:.2f}, {row['Random_NB_Q3']:.2f}] (n={int(row['Random_NB_N'])})")
            print(f"    Random DB: Median={row['Random_DB_Median']:.2f}% [{row['Random_DB_Q1']:.2f}, {row['Random_DB_Q3']:.2f}] (n={int(row['Random_DB_N'])})")
            print(f"    NB vs DB: p_raw={row['p_NB_vs_DB_raw']:.6f}, p_holm={row['p_NB_vs_DB_holm']:.6f} ({row['NB_vs_DB_Sig']})")
            print(f"    NB vs Random NB: p_raw={row['p_NB_vs_Random_NB_raw']:.6f}, p_holm={row['p_NB_vs_Random_NB_holm']:.6f} ({row['NB_vs_Random_NB_Sig']})")
            print(f"    DB vs Random DB: p_raw={row['p_DB_vs_Random_DB_raw']:.6f}, p_holm={row['p_DB_vs_Random_DB_holm']:.6f} ({row['DB_vs_Random_DB_Sig']})")

def main():
    parser = argparse.ArgumentParser(description='Figure 4D: Adaptive Behavior by Role Analysis')
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
    
    all_files = nb_files + db_files + random_nb_files + random_db_files
    
    print(f"Found {len(nb_files)} NB, {len(db_files)} DB, {len(random_nb_files)} Random NB, {len(random_db_files)} Random DB files")
    
    if not all_files:
        print("Error: No CSV files found")
        return
    
    # Extract data
    df_combined = extract_adaptive_behaviors(all_files)
    
    if df_combined.empty:
        print("Error: No data extracted")
        return
    
    print(f"\nTotal runs extracted: {len(df_combined)}")
    print(f"Conditions: {df_combined['condition'].unique()}")
    print(f"Roles: {df_combined['role'].unique()}")
    
    # Perform statistical tests
    stats_df = perform_statistical_tests(df_combined)
    
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
    raw_path = os.path.join(args.output, 'figure4d_raw_behavior_percentages.csv')
    raw_df.to_csv(raw_path, index=False)
    print(f"\nExported raw data to {raw_path}")
    
    # Export statistics
    stats_path = os.path.join(args.output, 'figure4d_stats_kruskal.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Exported statistics to {stats_path}")

if __name__ == '__main__':
    main()
