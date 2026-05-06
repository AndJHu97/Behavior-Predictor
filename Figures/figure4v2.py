#!/usr/bin/env python3
"""
Figure 4v2: Compositional Behavior Analysis by Agent Type and Adversity Condition

This script analyzes the composition of three behavioral clusters (Affiliative,
Internalizing, Externalizing) across adversity conditions and agent types (DB vs NB).

Clusters:
- Affiliative: Healthy friendliness, Help-seeking, Protective behavior
- Internalizing: Hypervigilant withdrawal, Adaptive avoidance, Learned Helplessness
- Externalizing: Misdirected aggression, Relational aggression, Dangerous trust

Denominator: Sum of all 9 complex behaviors (cluster sum to 100% within each run).

Statistical Analysis:
- Primary: Kruskal-Wallis (within agent type across conditions) + Dunn's test (post-hoc)
- Secondary: Mann-Whitney U (between agent types within condition)
- Optional: Dirichlet regression (handles compositional data dependency)
"""

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, sem

try:
    from statsmodels.stats.multitest import multipletests
    HAS_MULTIPLETESTS = True
except ImportError:
    HAS_MULTIPLETESTS = False

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

# Behavior to cluster mappings
AFFILIATIVE = ["healthy_friendliness", "community_trusting_vulnerability", "protective_behavior"]
INTERNALIZING = ["fearful_withdrawn_relationship", "willingness_to_flee", "learned_helplessness"]
EXTERNALIZING = ["bully_behavior", "aggressive_withdrawn_relationship", "dangerous_trust"]

ALL_BEHAVIORS = AFFILIATIVE + INTERNALIZING + EXTERNALIZING

CLUSTER_NAMES = ["Affiliative", "Internalizing", "Externalizing"]
CLUSTER_BEHAVIORS = [AFFILIATIVE, INTERNALIZING, EXTERNALIZING]

CONDITIONS_MAP = {
    "No ACE": ["LowThreat"],
    "Moderate": ["ModerateThreat"],
    "ACE": ["HighThreat"],
}

AGENT_TYPES = ["DB", "NB"]

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "Figures")
OUTPUT_DIR = FIGURES_DIR


@dataclass
class StatResult:
    cluster: str
    agent_type: str
    test_type: str
    comparison: str
    statistic: float
    p_value: float
    significant: bool


def flatten_globs(patterns: List[str], include_series: bool = False) -> List[str]:
    """Recursively glob file patterns, excluding archive and Figures folders."""
    files: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    
    cleaned: List[str] = []
    for path in dict.fromkeys(files):
        lower = path.lower()
        if "archive" in lower or os.path.sep + "figures" + os.path.sep in lower:
            continue
        if not include_series and ("_series_" in lower or "series.csv" in lower):
            continue
        cleaned.append(path)
    return cleaned


def infer_agent_type(path: str) -> Optional[str]:
    """Infer agent type (DB/NB/Random) from filename."""
    lower = path.lower()
    if "random" in lower:
        if "nb" in lower:
            return "Random NB"
        elif "db" in lower:
            return "Random DB"
        return "Random"
    elif "nb" in lower:
        return "NB"
    elif "db" in lower:
        return "DB"
    return None


def infer_condition(path: str) -> Optional[str]:
    """Infer condition from filename."""
    for condition, patterns in CONDITIONS_MAP.items():
        for pattern in patterns:
            if pattern in path:
                return condition
    return None


def find_run_header(rows: List[List[str]]) -> int:
    """Find the row index where 'Run,' header starts."""
    for i, row in enumerate(rows):
        if row and row[0].startswith("Run"):
            return i
    return -1


def resolve_columns(header: List[str], needed_cols: List[str]) -> Dict[str, int]:
    """Resolve column indices for needed behavior columns."""
    resolved = {}
    for col in needed_cols:
        if col in header:
            resolved[col] = header.index(col)
    return resolved


def parse_run_rows(csv_path: str, needed_cols: List[str]) -> pd.DataFrame:
    """Parse run-level rows from CSV file."""
    with open(csv_path, "r") as f:
        lines = f.readlines()
    
    header_idx = find_run_header([[cell.strip() for cell in line.split(",")] for line in lines])
    if header_idx < 0:
        return pd.DataFrame()
    
    header = [cell.strip() for cell in lines[header_idx].split(",")]
    col_indices = resolve_columns(header, needed_cols)
    if not col_indices:
        return pd.DataFrame()
    
    rows_data = []
    for line in lines[header_idx + 1:]:
        cells = [cell.strip() for cell in line.split(",")]
        if not cells or not cells[0]:
            continue
        row_dict = {}
        for col_name, col_idx in col_indices.items():
            try:
                row_dict[col_name] = float(cells[col_idx]) if col_idx < len(cells) else 0.0
            except (ValueError, IndexError):
                row_dict[col_name] = 0.0
        rows_data.append(row_dict)
    
    return pd.DataFrame(rows_data)


def load_all_runs(files: List[str]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Load run-level behavior counts and compute cluster percentages."""
    all_data = []
    skipped = []
    
    for csv_path in files:
        agent_type = infer_agent_type(csv_path)
        condition = infer_condition(csv_path)
        
        if not agent_type or not condition or "Random" in agent_type:
            skipped.append((csv_path, "Could not infer agent type or condition, or is Random"))
            continue
        
        df_runs = parse_run_rows(csv_path, ALL_BEHAVIORS)
        if df_runs.empty:
            skipped.append((csv_path, "No run rows parsed"))
            continue
        
        # Compute cluster sums
        for cluster_behaviors in CLUSTER_BEHAVIORS:
            cluster_name = CLUSTER_NAMES[CLUSTER_BEHAVIORS.index(cluster_behaviors)]
            present_behaviors = [b for b in cluster_behaviors if b in df_runs.columns]
            if present_behaviors:
                df_runs[f"{cluster_name}_count"] = df_runs[present_behaviors].sum(axis=1)
        
        # Compute total and percentages
        df_runs["total_behaviors"] = df_runs[ALL_BEHAVIORS].sum(axis=1)
        
        for cluster_name in CLUSTER_NAMES:
            df_runs[f"{cluster_name}_pct"] = (
                df_runs[f"{cluster_name}_count"] / df_runs["total_behaviors"].replace(0, np.nan)
            ) * 100
            df_runs[f"{cluster_name}_pct"] = df_runs[f"{cluster_name}_pct"].fillna(0)
        
        # Add metadata
        df_runs["agent_type"] = agent_type
        df_runs["condition"] = condition
        df_runs["source_file"] = os.path.basename(csv_path)
        
        all_data.append(df_runs)
    
    if not all_data:
        return pd.DataFrame(), skipped
    
    return pd.concat(all_data, ignore_index=True), skipped


def dunn_test_posthoc(data: np.ndarray, groups: np.ndarray) -> List[Dict]:
    """Perform Dunn's test (pairwise comparisons after Kruskal-Wallis) with Holm correction."""
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return []
    
    # Use pingouin if available, else fallback to manual calculation
    if HAS_PINGOUIN:
        try:
            df_input = pd.DataFrame({
                'value': data,
                'group': groups
            })
            result = pg.pairwise_dunn(
                data=df_input,
                dv='value',
                between='group'
            )
            
            results = []
            for idx, row in result.iterrows():
                results.append({
                    "group1": row['A'],
                    "group2": row['B'],
                    "p_value": row['p-adjust'],
                    "statistic": row['Z-val'],
                })
            return results
        except Exception:
            pass
    
    # Fallback: manual pairwise Mann-Whitney with Holm correction
    results = []
    p_values = []
    comparisons = []
    
    for i, g1 in enumerate(unique_groups):
        for g2 in unique_groups[i+1:]:
            data1 = data[groups == g1]
            data2 = data[groups == g2]
            if len(data1) > 0 and len(data2) > 0:
                u_stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                p_values.append(p_val)
                comparisons.append((g1, g2, u_stat))
    
    if p_values:
        # Apply Holm correction
        if HAS_MULTIPLETESTS:
            rejected, p_corrected, _, _ = multipletests(p_values, method='holm')
            for (g1, g2, u_stat), p_corr in zip(comparisons, p_corrected):
                results.append({
                    "group1": g1,
                    "group2": g2,
                    "p_value": p_corr,
                    "statistic": u_stat,
                })
        else:
            # Manual Holm correction if statsmodels not available
            sorted_indices = np.argsort(p_values)
            p_corrected = np.array(p_values)[sorted_indices]
            p_corrected = p_corrected * np.arange(len(p_corrected), 0, -1)
            p_corrected = np.minimum(p_corrected, 1.0)
            
            result_array = np.empty((len(comparisons),), dtype=object)
            for idx, orig_idx in enumerate(sorted_indices):
                result_array[orig_idx] = p_corrected[idx]
            
            for (g1, g2, u_stat), p_corr in zip(comparisons, result_array):
                results.append({
                    "group1": g1,
                    "group2": g2,
                    "p_value": p_corr,
                    "statistic": u_stat,
                })
    
    return results


def perform_statistical_tests(df: pd.DataFrame) -> List[StatResult]:
    """Perform Kruskal-Wallis, Dunn, and Mann-Whitney U tests."""
    results = []
    
    conditions = sorted(df["condition"].unique())
    
    # Primary: Within each agent type, across conditions (Kruskal-Wallis + Dunn)
    for agent_type in AGENT_TYPES:
        agent_df = df[df["agent_type"] == agent_type]
        
        for cluster_name in CLUSTER_NAMES:
            pct_col = f"{cluster_name}_pct"
            
            # Prepare data for Kruskal-Wallis
            groups_data = []
            groups_labels = []
            
            condition_order_subset = ["No ACE", "Moderate", "ACE"]
            for condition in condition_order_subset:
                if condition in agent_df["condition"].unique():
                    cond_data = agent_df[agent_df["condition"] == condition][pct_col].dropna().values
                    groups_data.extend(cond_data)
                    groups_labels.extend([condition] * len(cond_data))
            
            if len(np.unique(groups_labels)) < 2:
                continue
            
            # Kruskal-Wallis test
            unique_conds = sorted(set(groups_labels))
            group_arrays = [
                np.array(groups_data)[np.array(groups_labels) == cond]
                for cond in unique_conds
            ]
            
            if all(len(g) > 0 for g in group_arrays):
                h_stat, p_kw = kruskal(*group_arrays)
                
                results.append(
                    StatResult(
                        cluster=cluster_name,
                        agent_type=agent_type,
                        test_type="Kruskal-Wallis",
                        comparison=f"Across {len(unique_conds)} conditions",
                        statistic=h_stat,
                        p_value=p_kw,
                        significant=(p_kw < 0.05),
                    )
                )
                
                # Post-hoc Dunn test if significant
                if p_kw < 0.05:
                    try:
                        dunn_results = dunn_test_posthoc(np.array(groups_data), np.array(groups_labels))
                        for dunn_res in dunn_results:
                            results.append(
                                StatResult(
                                    cluster=cluster_name,
                                    agent_type=agent_type,
                                    test_type="Dunn (Holm)",
                                    comparison=f"{dunn_res['group1']} vs {dunn_res['group2']}",
                                    statistic=dunn_res.get("statistic", np.nan),
                                    p_value=dunn_res["p_value"],
                                    significant=(dunn_res["p_value"] < 0.05),
                                )
                            )
                    except Exception:
                        pass
    
    # Secondary: Between agent types within each condition (Mann-Whitney U)
    condition_order_subset = ["No ACE", "Moderate", "ACE"]
    conditions_for_mw = [c for c in condition_order_subset if c in df["condition"].unique()]
    for condition in conditions_for_mw:
        for cluster_name in CLUSTER_NAMES:
            pct_col = f"{cluster_name}_pct"
            
            db_data = df[(df["agent_type"] == "DB") & (df["condition"] == condition)][pct_col].dropna().values
            nb_data = df[(df["agent_type"] == "NB") & (df["condition"] == condition)][pct_col].dropna().values
            
            if len(db_data) > 0 and len(nb_data) > 0:
                u_stat, p_mw = mannwhitneyu(db_data, nb_data, alternative="two-sided")
                
                results.append(
                    StatResult(
                        cluster=cluster_name,
                        agent_type="DB vs NB",
                        test_type="Mann-Whitney U",
                        comparison=condition,
                        statistic=u_stat,
                        p_value=p_mw,
                        significant=(p_mw < 0.05),
                    )
                )
    
    return results


def plot_grouped_bars(df: pd.DataFrame, output_path: str) -> None:
    """Create side-by-side grouped bar charts for DB and NB."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # Define custom condition order: No ACE -> Moderate -> ACE
    condition_order = ["No ACE", "Moderate", "ACE"]
    conditions = [c for c in condition_order if c in df["condition"].unique()]
    cluster_colors = {"Affiliative": "#2ecc71", "Internalizing": "#e74c3c", "Externalizing": "#3498db"}
    
    x_pos = np.arange(len(conditions))
    bar_width = 0.25
    
    for ax_idx, agent_type in enumerate(AGENT_TYPES):
        ax = axes[ax_idx]
        agent_df = df[df["agent_type"] == agent_type]
        
        for cluster_idx, cluster_name in enumerate(CLUSTER_NAMES):
            pct_col = f"{cluster_name}_pct"
            
            means = []
            sems = []
            
            for condition in conditions:
                cond_data = agent_df[agent_df["condition"] == condition][pct_col].dropna().values
                if len(cond_data) > 0:
                    means.append(np.mean(cond_data))
                    sems.append(sem(cond_data))
                else:
                    means.append(0)
                    sems.append(0)
            
            x_offset = x_pos + (cluster_idx - 1) * bar_width
            ax.bar(
                x_offset,
                means,
                bar_width,
                label=cluster_name,
                color=cluster_colors[cluster_name],
                alpha=0.8,
                edgecolor="black",
                linewidth=1.0,
                yerr=sems,
                capsize=5,
            )
        
        ax.set_xlabel("Adversity Condition", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Percentage (%)", fontsize=12, fontweight="bold")
        ax.set_title(f"{agent_type} Agents", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions, fontsize=11)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, 100)
    
    fig.suptitle("Figure 4v2: Behavioral Cluster Composition by Agent Type and Condition", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4v2: Compositional Behavior Analysis")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[os.path.join(WORKSPACE_ROOT, "**", "multiple_runs*Threat*.csv")],
        help="Input CSV files or glob patterns",
    )
    parser.add_argument(
        "--include-series",
        action="store_true",
        help="Include CSVs with 'series' in their filename",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    files = flatten_globs(args.inputs, include_series=args.include_series)
    
    if not files:
        raise SystemExit("No matching run CSV files found.")
    
    raw_df, skipped = load_all_runs(files)
    if raw_df.empty:
        raise SystemExit("No valid run-level rows parsed from the input files.")
    
    # Perform statistical tests
    stat_results = perform_statistical_tests(raw_df)
    
    # Save outputs
    fig_path = os.path.join(args.output_dir, "figure4v2_grouped_bars.png")
    raw_path = os.path.join(args.output_dir, "figure4v2_raw_run_percentages.csv")
    stats_path = os.path.join(args.output_dir, "figure4v2_stats.csv")
    summary_path = os.path.join(args.output_dir, "figure4v2_summary.csv")
    
    # Plot
    plot_grouped_bars(raw_df, fig_path)
    
    # Export raw run percentages
    export_cols = ["agent_type", "condition", "source_file"]
    for cluster_name in CLUSTER_NAMES:
        export_cols.append(f"{cluster_name}_pct")
    raw_df[export_cols].to_csv(raw_path, index=False)
    print(f"Saved raw run percentages to {raw_path}")
    
    # Export statistics
    stat_df = pd.DataFrame([
        {
            "cluster": r.cluster,
            "agent_type": r.agent_type,
            "test_type": r.test_type,
            "comparison": r.comparison,
            "statistic": r.statistic,
            "p_value": r.p_value,
            "significant": r.significant,
        }
        for r in stat_results
    ])
    stat_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to {stats_path}")
    
    # Compute summary table (mean percentages by agent and condition)
    summary_rows = []
    condition_order_subset = ["No ACE", "Moderate", "ACE"]
    conditions_for_summary = [c for c in condition_order_subset if c in raw_df["condition"].unique()]
    for agent_type in AGENT_TYPES:
        for condition in conditions_for_summary:
            subset = raw_df[(raw_df["agent_type"] == agent_type) & (raw_df["condition"] == condition)]
            summary_rows.append({
                "agent_type": agent_type,
                "condition": condition,
                "n_runs": len(subset),
                "Affiliative_mean": subset["Affiliative_pct"].mean(),
                "Affiliative_std": subset["Affiliative_pct"].std(),
                "Internalizing_mean": subset["Internalizing_pct"].mean(),
                "Internalizing_std": subset["Internalizing_pct"].std(),
                "Externalizing_mean": subset["Externalizing_pct"].mean(),
                "Externalizing_std": subset["Externalizing_pct"].std(),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to {summary_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("Figure 4v2: Compositional Behavior Analysis Summary")
    print("=" * 100)
    print(f"Total runs analyzed: {len(raw_df)}")
    print(f"Agent types: {sorted(raw_df['agent_type'].unique())}")
    print(f"Conditions: {sorted(raw_df['condition'].unique())}")
    print(f"\nSummary by Agent Type and Condition:")
    print(summary_df.to_string(index=False))
    
    if stat_df.shape[0] > 0:
        print(f"\n\nStatistical Test Results ({len(stat_df)} tests):")
        print(stat_df.to_string(index=False))


if __name__ == "__main__":
    main()
