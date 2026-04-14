#!/usr/bin/env python3
"""
Figure 5D: Adaptive Behavior Emergence Analysis - Split by Role (NB vs DB vs Random)
Uses a global 75th percentile emergence threshold computed on pooled run-level percentages.
Outputs role-split emergence probabilities, raw run-level presence data, thresholds, and stats.
"""

import argparse
import csv
import glob
import os
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(invalid='ignore')

CONDITIONS = ['No ACE', 'Moderate', 'ACE']

ADAPTIVE_COMPLEX_BEHAVIORS = [
    'protective_behavior',
    'healthy_friendliness',
    'willingness_to_flee',
    'community_trusting_vulnerability',
    'hopefulness',
]

MALADAPTIVE_COMPLEX_BEHAVIORS = [
    'fearful_withdrawn_relationship',
    'dangerous_trust',
    'bully_behavior',
    'aggressive_withdrawn_relationship',
    'cynical',
    'learned_helplessness',
]

TOTAL_COMPLEX_BEHAVIOR_COLUMNS = list(dict.fromkeys(ADAPTIVE_COMPLEX_BEHAVIORS + MALADAPTIVE_COMPLEX_BEHAVIORS))
EXPECTANCY_DENOM_COLUMNS = ['hopefulness', 'cynical', 'learned_helplessness']

ADAPTIVE_EMERGENCE_SPECS = [
    ('protective_behavior', 'Protective Behavior', 'protective_behavior', 'total_complex_behaviors'),
    ('healthy_friendliness', 'Healthy Friendliness', 'healthy_friendliness', 'total_complex_behaviors'),
    ('willingness_to_flee', 'Adaptive Avoidance', 'willingness_to_flee', 'total_complex_behaviors'),
    ('community_trusting_vulnerability', 'Help-Seeking Vulnerability', 'community_trusting_vulnerability', 'total_complex_behaviors'),
    ('hopefulness', 'Positive Expectancy', 'hopefulness', 'total_expectancies'),
]


def flatten_globs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    return list(dict.fromkeys(files))


def infer_condition(path: str) -> str:
    name = os.path.basename(path).lower()
    if 'highthreat' in name or '_ace' in name:
        return 'ACE'
    if 'moderatethreat' in name or 'moderate' in name:
        return 'Moderate'
    if 'lowthreat' in name or 'default' in name or 'noace' in name:
        return 'No ACE'
    return ''


def parse_role(path: str) -> str:
    name = os.path.basename(path).lower()
    if 'random' in name:
        if 'nb' in name:
            return 'Random NB'
        if 'db' in name:
            return 'Random DB'
        return 'Random'
    if 'nb' in name:
        return 'NB'
    if 'db' in name:
        return 'DB'
    return ''


def parse_run_rows(csv_path: str, needed_cols: List[str]) -> List[Tuple[int, Dict[str, float]]]:
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip().lower() == 'run':
            header_set = {cell.strip() for cell in row}
            if set(needed_cols + ['Run']).issubset(header_set):
                header_idx = i
                break

    if header_idx is None:
        return []

    header = [cell.strip() for cell in rows[header_idx]]
    idx = {name: header.index(name) for name in needed_cols + ['Run']}

    parsed: List[Tuple[int, Dict[str, float]]] = []
    for row in rows[header_idx + 1:]:
        if not row:
            break
        first = row[0].strip().lower()
        if first in {'statistic', 'total', 'mean', 'std', 'min', 'max'}:
            break
        try:
            run_id = int(float(row[idx['Run']]))
            values = {col: float(row[idx[col]]) for col in needed_cols}
        except (ValueError, IndexError):
            continue
        parsed.append((run_id, values))
    return parsed


def holm_adjust(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(p_values)
    out = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = (m - rank) * p
        running_max = max(running_max, adjusted)
        out[orig_idx] = min(1.0, running_max)
    return out


def extract_emergence_data(csv_files: List[str]) -> pd.DataFrame:
    all_data = []

    for csv_file in csv_files:
        condition = infer_condition(csv_file)
        role = parse_role(csv_file)
        if not condition or not role:
            print(f'Skipping {csv_file}: could not determine condition or role')
            continue

        try:
            parsed = parse_run_rows(csv_file, TOTAL_COMPLEX_BEHAVIOR_COLUMNS)
            if not parsed:
                print(f'No run data in {csv_file}')
                continue

            df_runs = pd.DataFrame([{'Run': run_id, **values} for run_id, values in parsed])
            for col in TOTAL_COMPLEX_BEHAVIOR_COLUMNS:
                df_runs[col] = pd.to_numeric(df_runs[col], errors='coerce')

            if not all(col in df_runs.columns for col in TOTAL_COMPLEX_BEHAVIOR_COLUMNS):
                print(f'Missing behavior columns in {csv_file}')
                continue

            df_runs['total_complex_behaviors'] = df_runs[TOTAL_COMPLEX_BEHAVIOR_COLUMNS].sum(axis=1)
            df_runs['total_expectancies'] = df_runs[EXPECTANCY_DENOM_COLUMNS].sum(axis=1)

            for behavior_key, behavior_name, numerator_col, denom_col in ADAPTIVE_EMERGENCE_SPECS:
                denom = df_runs[denom_col].replace(0, np.nan)
                df_runs[f'{behavior_key}_pct'] = (df_runs[numerator_col] / denom) * 100.0
                df_runs[f'{behavior_key}_pct'] = df_runs[f'{behavior_key}_pct'].fillna(0)

            df_runs['source_file'] = os.path.basename(csv_file)
            df_runs['condition'] = condition
            df_runs['role'] = role
            all_data.append(df_runs)
            print(f'Extracted {len(df_runs)} runs from {csv_file} ({condition}, {role})')
        except Exception as exc:
            print(f'Error processing {csv_file}: {exc}')
            continue

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def apply_global_thresholds(df_combined: pd.DataFrame, threshold_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_combined = df_combined.copy()
    thresholds_rows = []

    pooled = df_combined[df_combined['role'].isin(['NB', 'DB'])]
    if pooled.empty:
        pooled = df_combined

    for behavior_key, behavior_name, _numerator_col, _denom_col in ADAPTIVE_EMERGENCE_SPECS:
        threshold = float(np.percentile(pooled[f'{behavior_key}_pct'].values, threshold_pct))
        df_combined[f'{behavior_key}_threshold'] = threshold
        df_combined[f'{behavior_key}_present'] = (df_combined[f'{behavior_key}_pct'] >= threshold).astype(int)
        thresholds_rows.append({
            'behavior_key': behavior_key,
            'behavior': behavior_name,
            'threshold_percentile': threshold_pct,
            'threshold_percent': threshold,
            'threshold_pool': 'NB+DB',
        })

    return df_combined, pd.DataFrame(thresholds_rows)


def safe_chi_square(table: np.ndarray) -> Tuple[float, float, float]:
    if np.any(table.sum(axis=1) == 0) or np.any(table.sum(axis=0) == 0):
        return np.nan, np.nan, np.nan
    try:
        chi2, p_value, dof, _ = chi2_contingency(table)
        n_total = table.sum()
        k = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n_total * k)) if n_total > 0 and k > 0 else np.nan
        return chi2, p_value, cramers_v
    except ValueError:
        return np.nan, np.nan, np.nan


def perform_statistical_tests(df_combined: pd.DataFrame) -> pd.DataFrame:
    results = []
    for behavior_key, behavior_name, _numerator_col, _denom_col in ADAPTIVE_EMERGENCE_SPECS:
        present_col = f'{behavior_key}_present'
        if present_col not in df_combined.columns:
            print(f'Skipping {behavior_key}: column not found')
            continue

        for condition in CONDITIONS:
            nb_sub = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'NB')]
            db_sub = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'DB')]
            random_nb_sub = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'Random NB')]
            random_db_sub = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'Random DB')]

            if nb_sub.empty or db_sub.empty or random_nb_sub.empty or random_db_sub.empty:
                continue

            nb_present_count = int(nb_sub[present_col].sum())
            db_present_count = int(db_sub[present_col].sum())
            random_nb_present_count = int(random_nb_sub[present_col].sum())
            random_db_present_count = int(random_db_sub[present_col].sum())

            nb_n = len(nb_sub)
            db_n = len(db_sub)
            random_nb_n = len(random_nb_sub)
            random_db_n = len(random_db_sub)

            nb_prob = (nb_present_count / nb_n) * 100 if nb_n else np.nan
            db_prob = (db_present_count / db_n) * 100 if db_n else np.nan
            random_nb_prob = (random_nb_present_count / random_nb_n) * 100 if random_nb_n else np.nan
            random_db_prob = (random_db_present_count / random_db_n) * 100 if random_db_n else np.nan

            chi2, p_global, cramers_v = safe_chi_square(np.array([
                [nb_present_count, nb_n - nb_present_count],
                [db_present_count, db_n - db_present_count],
                [random_nb_present_count, random_nb_n - random_nb_present_count],
                [random_db_present_count, random_db_n - random_db_present_count],
            ], dtype=float))

            _, p_nb_db, _ = safe_chi_square(np.array([[nb_present_count, nb_n - nb_present_count], [db_present_count, db_n - db_present_count]], dtype=float))
            _, p_nb_random, _ = safe_chi_square(np.array([[nb_present_count, nb_n - nb_present_count], [random_nb_present_count, random_nb_n - random_nb_present_count]], dtype=float))
            _, p_db_random, _ = safe_chi_square(np.array([[db_present_count, db_n - db_present_count], [random_db_present_count, random_db_n - random_db_present_count]], dtype=float))

            p_raw = np.array([p_nb_db, p_nb_random, p_db_random], dtype=float)
            valid_mask = ~np.isnan(p_raw)
            p_holm = np.full_like(p_raw, np.nan, dtype=float)
            if np.any(valid_mask):
                p_holm[valid_mask] = holm_adjust(p_raw[valid_mask].tolist())

            results.append({
                'Behavior': behavior_name,
                'Condition': condition,
                'NB_Probability': nb_prob,
                'DB_Probability': db_prob,
                'Random_NB_Probability': random_nb_prob,
                'Random_DB_Probability': random_db_prob,
                'NB_N': nb_n,
                'DB_N': db_n,
                'Random_NB_N': random_nb_n,
                'Random_DB_N': random_db_n,
                'NB_Present': nb_present_count,
                'DB_Present': db_present_count,
                'Random_NB_Present': random_nb_present_count,
                'Random_DB_Present': random_db_present_count,
                'Chi2': chi2,
                'p_global': p_global,
                'Cramers_V': cramers_v,
                'NB_vs_DB_p_raw': p_nb_db,
                'NB_vs_DB_p_holm': p_holm[0],
                'NB_vs_Random_NB_p_raw': p_nb_random,
                'NB_vs_Random_NB_p_holm': p_holm[1],
                'DB_vs_Random_DB_p_raw': p_db_random,
                'DB_vs_Random_DB_p_holm': p_holm[2],
            })

    return pd.DataFrame(results)


def create_visualization(df_combined: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    behaviors = [label for _, label, _, _ in ADAPTIVE_EMERGENCE_SPECS]
    conditions = CONDITIONS

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    rng = np.random.default_rng(42)
    cond_colors = {'No ACE': '#1f77b4', 'Moderate': '#ff7f0e', 'ACE': '#d62728'}
    role_offsets = {'NB': -0.18, 'DB': 0.18}
    bar_width = 0.32

    for idx, behavior_name in enumerate(behaviors):
        ax = axes[idx]
        behavior_var = [k for k, v, _, _ in ADAPTIVE_EMERGENCE_SPECS if v == behavior_name][0]
        x_pos = np.arange(len(conditions), dtype=float)

        for i, condition in enumerate(conditions):
            nb_vals = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'NB')][f'{behavior_var}_present'].dropna().values
            db_vals = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'DB')][f'{behavior_var}_present'].dropna().values
            random_nb_vals = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'Random NB')][f'{behavior_var}_present'].dropna().values
            random_db_vals = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == 'Random DB')][f'{behavior_var}_present'].dropna().values

            if nb_vals.size > 0:
                nb_prob = float(np.mean(nb_vals) * 100.0)
                nb_n = len(nb_vals)
                nb_ci = 1.96 * np.sqrt(max((nb_prob / 100.0) * (1.0 - (nb_prob / 100.0)), 0.0) / nb_n) * 100.0
                nb_q1 = max(0.0, nb_prob - nb_ci)
                nb_q3 = min(100.0, nb_prob + nb_ci)
                nb_x = x_pos[i] + role_offsets['NB']
                ax.bar(nb_x, nb_prob, bar_width, color=cond_colors[condition], alpha=0.75, edgecolor='black', linewidth=1.2)
                ax.errorbar(nb_x, nb_prob, yerr=[[nb_prob - nb_q1], [nb_q3 - nb_prob]], fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)
                nb_jitter = rng.normal(0, bar_width * 0.12, size=nb_vals.size)
                ax.scatter(nb_x + nb_jitter, nb_vals * 100.0, alpha=0.35, s=18, color=cond_colors[condition], edgecolors='black', linewidth=0.4)

            if db_vals.size > 0:
                db_prob = float(np.mean(db_vals) * 100.0)
                db_n = len(db_vals)
                db_ci = 1.96 * np.sqrt(max((db_prob / 100.0) * (1.0 - (db_prob / 100.0)), 0.0) / db_n) * 100.0
                db_q1 = max(0.0, db_prob - db_ci)
                db_q3 = min(100.0, db_prob + db_ci)
                db_x = x_pos[i] + role_offsets['DB']
                ax.bar(db_x, db_prob, bar_width, color=cond_colors[condition], alpha=0.75, edgecolor='black', linewidth=1.2, hatch='//')
                ax.errorbar(db_x, db_prob, yerr=[[db_prob - db_q1], [db_q3 - db_prob]], fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)
                db_jitter = rng.normal(0, bar_width * 0.12, size=db_vals.size)
                ax.scatter(db_x + db_jitter, db_vals * 100.0, alpha=0.35, s=18, color=cond_colors[condition], edgecolors='black', linewidth=0.4)

            if random_nb_vals.size > 0:
                y_nb_base = float(np.mean(random_nb_vals) * 100.0)
                x_left = x_pos[i] + role_offsets['NB'] - (bar_width * 0.46)
                x_right = x_pos[i] + role_offsets['NB'] + (bar_width * 0.46)
                ax.hlines(y_nb_base, x_left, x_right, colors='#7f7f7f', linestyles='--', linewidth=2.0)

            if random_db_vals.size > 0:
                y_db_base = float(np.mean(random_db_vals) * 100.0)
                x_left = x_pos[i] + role_offsets['DB'] - (bar_width * 0.46)
                x_right = x_pos[i] + role_offsets['DB'] + (bar_width * 0.46)
                ax.hlines(y_db_base, x_left, x_right, colors='#7f7f7f', linestyles='-.', linewidth=2.0)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions, fontsize=11)
        ax.set_ylabel('% of runs where behavior is present', fontsize=11, fontweight='bold')
        ax.set_title(behavior_name, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)

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

    fig.delaxes(axes[5])
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'figure5d_adaptive_by_role.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved visualization to {output_path}')
    plt.close(fig)


def print_summary_table(stats_df: pd.DataFrame) -> None:
    print('\n' + '=' * 160)
    print('FIGURE 5D: ADAPTIVE EMERGENCE BY ROLE - Summary Statistics')
    print('=' * 160)

    for behavior in stats_df['Behavior'].unique():
        print(f'\n{behavior}:')
        behavior_data = stats_df[stats_df['Behavior'] == behavior]
        for condition in CONDITIONS:
            cond_data = behavior_data[behavior_data['Condition'] == condition]
            if cond_data.empty:
                continue
            row = cond_data.iloc[0]
            print(f"\n  {condition}:")
            print(f"    NB:     {row['NB_Present']}/{row['NB_N']} present ({row['NB_Probability']:.2f}%)")
            print(f"    DB:     {row['DB_Present']}/{row['DB_N']} present ({row['DB_Probability']:.2f}%)")
            print(f"    Random NB: {row['Random_NB_Present']}/{row['Random_NB_N']} present ({row['Random_NB_Probability']:.2f}%)")
            print(f"    Random DB: {row['Random_DB_Present']}/{row['Random_DB_N']} present ({row['Random_DB_Probability']:.2f}%)")
            print(f"    Chi-square: chi2={row['Chi2']:.2f}, p={row['p_global']:.6f}")
            print(f"    NB vs DB p_raw={row['NB_vs_DB_p_raw']:.6f}, p_holm={row['NB_vs_DB_p_holm']:.6f}")
            print(f"    NB vs Random NB p_raw={row['NB_vs_Random_NB_p_raw']:.6f}, p_holm={row['NB_vs_Random_NB_p_holm']:.6f}")
            print(f"    DB vs Random DB p_raw={row['DB_vs_Random_DB_p_raw']:.6f}, p_holm={row['DB_vs_Random_DB_p_holm']:.6f}")


def write_csv_with_fallback(df: pd.DataFrame, target_path: str) -> str:
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        root, ext = os.path.splitext(target_path)
        fallback = f'{root}_updated{ext}'
        df.to_csv(fallback, index=False)
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description='Figure 5D: Adaptive Emergence by Role Analysis')
    parser.add_argument('--NB', nargs='*', default=['multiple_runs_NB_*Threat*.csv'], help='NB file pattern(s)')
    parser.add_argument('--DB', nargs='*', default=['multiple_runs_DB_*Threat*.csv'], help='DB file pattern(s)')
    parser.add_argument('--RandomNB', nargs='*', default=['multiple_runs_random_NB_*Threat*.csv'], help='Random NB file pattern(s)')
    parser.add_argument('--RandomDB', nargs='*', default=['multiple_runs_random_DB_*Threat*.csv'], help='Random DB file pattern(s)')
    parser.add_argument('--threshold', type=float, default=75.0, help='Global percentile threshold for emergence')
    parser.add_argument('--output', type=str, default=os.path.join('FinalProject', 'figure5'), help='Output directory')

    args = parser.parse_args()

    nb_files = sorted(flatten_globs(args.NB))
    db_files = sorted(flatten_globs(args.DB))
    random_nb_files = sorted(flatten_globs(args.RandomNB))
    random_db_files = sorted(flatten_globs(args.RandomDB))
    all_files = nb_files + db_files + random_nb_files + random_db_files

    print(f'Found {len(nb_files)} NB, {len(db_files)} DB, {len(random_nb_files)} Random NB, {len(random_db_files)} Random DB files')
    if not all_files:
        print('Error: No CSV files found')
        return

    df_combined = extract_emergence_data(all_files)
    if df_combined.empty:
        print('Error: No data extracted')
        return

    df_combined, thresholds_df = apply_global_thresholds(df_combined, args.threshold)

    print(f'\nTotal runs extracted: {len(df_combined)}')
    print(f"Conditions: {df_combined['condition'].unique()}")
    print(f"Roles: {df_combined['role'].unique()}")

    stats_df = perform_statistical_tests(df_combined)
    print_summary_table(stats_df)
    create_visualization(df_combined, args.output)

    os.makedirs(args.output, exist_ok=True)
    raw_export = []
    for behavior_key, behavior_name, numerator_col, denom_col in ADAPTIVE_EMERGENCE_SPECS:
        pct_col = f'{behavior_key}_pct'
        for idx, row in df_combined.iterrows():
            raw_export.append({
                'source_file': row['source_file'],
                'condition': row['condition'],
                'role': row['role'],
                'run': row.get('Run', idx),
                'behavior_key': behavior_key,
                'behavior': behavior_name,
                'behavior_count': row[numerator_col],
                'behavior_denominator': row[denom_col],
                'behavior_percent': row[pct_col],
                'threshold_percent': row.get(f'{behavior_key}_threshold', np.nan),
                'present': row[f'{behavior_key}_present'],
            })

    raw_df = pd.DataFrame(raw_export)
    raw_saved = write_csv_with_fallback(raw_df, os.path.join(args.output, 'figure5d_raw_emergence_presence.csv'))
    stats_saved = write_csv_with_fallback(stats_df, os.path.join(args.output, 'figure5d_stats_chi2.csv'))

    thresholds_saved = write_csv_with_fallback(thresholds_df, os.path.join(args.output, 'figure5d_thresholds.csv'))

    prob_rows = []
    for behavior_key, behavior_name, _numerator_col, _denom_col in ADAPTIVE_EMERGENCE_SPECS:
        for condition in CONDITIONS:
            for role in ['NB', 'DB', 'Random NB', 'Random DB']:
                sub = df_combined[(df_combined['condition'] == condition) & (df_combined['role'] == role)]
                n = len(sub)
                if n == 0:
                    continue
                present = int(sub[f'{behavior_key}_present'].sum())
                prob = (present / n) * 100.0
                se = np.sqrt(max((present / n) * (1.0 - (present / n)), 0.0) / n)
                ci = 1.96 * se * 100.0
                prob_rows.append({
                    'behavior_key': behavior_key,
                    'behavior': behavior_name,
                    'condition': condition,
                    'role': role,
                    'n_runs': n,
                    'n_present': present,
                    'probability_percent': prob,
                    'ci95_low_percent': max(0.0, prob - ci),
                    'ci95_high_percent': min(100.0, prob + ci),
                    'ci95_halfwidth_percent': ci,
                })
    prob_df = pd.DataFrame(prob_rows)
    prob_saved = write_csv_with_fallback(prob_df, os.path.join(args.output, 'figure5d_probability_by_condition_role.csv'))

    print(f'\nExported raw data to {raw_saved}')
    print(f'Exported thresholds to {thresholds_saved}')
    print(f'Exported probabilities to {prob_saved}')
    print(f'Exported statistics to {stats_saved}')


if __name__ == '__main__':
    main()
