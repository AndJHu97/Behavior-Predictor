#!/usr/bin/env python3
"""
Supplemental Figure: Allies Conditions compositional analysis (Standard, Strong, Very Strong).

This script compares three behavioral clusters across three ally conditions
(Standard Allies, Strong Allies, Very Strong Allies) and two agent types (DB vs NB).

Clusters:
- Affiliative: healthy_friendliness, community_trusting_vulnerability, protective_behavior
- Internalizing: fearful_withdrawn_relationship, willingness_to_flee, learned_helplessness
- Externalizing: bully_behavior, aggressive_withdrawn_relationship, dangerous_trust

The figure uses a three-tier x-axis:
- Top tier: Standard Allies / Strong Allies / Very Strong Allies
- Bottom tier: DB / NB / DB / NB / DB / NB

Statistical analysis:
- Kruskal-Wallis test for each cluster within each agent type across three ally conditions.
- Post-hoc Mann-Whitney U pairwise comparisons with Holm correction.
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, sem


# Behavior-to-cluster mappings.
AFFILIATIVE = ["healthy_friendliness", "community_trusting_vulnerability", "protective_behavior"]
INTERNALIZING = ["fearful_withdrawn_relationship", "willingness_to_flee", "learned_helplessness"]
EXTERNALIZING = ["bully_behavior", "aggressive_withdrawn_relationship", "dangerous_trust"]

ALL_BEHAVIORS = AFFILIATIVE + INTERNALIZING + EXTERNALIZING
CLUSTER_NAMES = ["Affiliative", "Internalizing", "Externalizing"]
CLUSTER_BEHAVIORS = [AFFILIATIVE, INTERNALIZING, EXTERNALIZING]
AGENT_TYPES = ["DB", "NB"]
ALLIES_CONDITIONS = ["Standard Allies", "Strong Allies", "Very Strong Allies"]

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "Figures")
OUTPUT_DIR = FIGURES_DIR


@dataclass
class StatResult:
    cluster: str
    agent_type: str
    comparison: str
    statistic: float
    p_value_raw: float
    p_value_holm: float
    significant: bool
    n_standard: int
    n_strong: int


def flatten_globs(patterns: List[str], include_series: bool = False) -> List[str]:
    """Glob input patterns and remove archive/Figures files."""
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
    """Infer agent type from filename."""
    lower = path.lower()
    if "random" in lower:
        if "nb" in lower:
            return "Random NB"
        if "db" in lower:
            return "Random DB"
        return "Random"
    if "nb" in lower:
        return "NB"
    if "db" in lower:
        return "DB"
    return None


def infer_allies_condition(path: str) -> Optional[str]:
    """Infer Standard Allies, Strong Allies, or Very Strong Allies from filename."""
    lower = path.lower()
    if "verystrongallies" in lower:
        return "Very Strong Allies"
    if "strongallies" in lower:
        return "Strong Allies"
    if "lowthreat" in lower:
        return "Standard Allies"
    return None


def find_run_header(rows: List[List[str]]) -> int:
    """Find the row index where the run table begins."""
    for i, row in enumerate(rows):
        if row and row[0].startswith("Run"):
            return i
    return -1


def resolve_columns(header: List[str], needed_cols: List[str]) -> Dict[str, int]:
    """Resolve column indices for the requested behavior columns."""
    resolved: Dict[str, int] = {}
    for col in needed_cols:
        if col in header:
            resolved[col] = header.index(col)
    return resolved


def parse_run_rows(csv_path: str, needed_cols: List[str]) -> pd.DataFrame:
    """Parse run-level rows from a CSV file."""
    with open(csv_path, "r", encoding="utf-8") as f:
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
    """Load run-level counts and compute cluster percentages for each run."""
    all_data = []
    skipped = []

    for csv_path in files:
        agent_type = infer_agent_type(csv_path)
        allies_condition = infer_allies_condition(csv_path)

        if not agent_type or not allies_condition or "Random" in agent_type:
            skipped.append((csv_path, "Could not infer agent type or allies condition, or file is random"))
            continue

        df_runs = parse_run_rows(csv_path, ALL_BEHAVIORS)
        if df_runs.empty:
            skipped.append((csv_path, "No run rows parsed"))
            continue

        for behavior in ALL_BEHAVIORS:
            if behavior not in df_runs.columns:
                df_runs[behavior] = 0.0

        for cluster_name, cluster_behaviors in zip(CLUSTER_NAMES, CLUSTER_BEHAVIORS):
            df_runs[f"{cluster_name}_count"] = df_runs[cluster_behaviors].sum(axis=1)

        df_runs["total_behaviors"] = df_runs[ALL_BEHAVIORS].sum(axis=1)

        for cluster_name in CLUSTER_NAMES:
            df_runs[f"{cluster_name}_pct"] = (
                df_runs[f"{cluster_name}_count"] / df_runs["total_behaviors"].replace(0, np.nan)
            ) * 100
            df_runs[f"{cluster_name}_pct"] = df_runs[f"{cluster_name}_pct"].fillna(0)

        df_runs["agent_type"] = agent_type
        df_runs["allies_condition"] = allies_condition
        df_runs["source_file"] = os.path.basename(csv_path)

        all_data.append(df_runs)

    if not all_data:
        return pd.DataFrame(), skipped

    return pd.concat(all_data, ignore_index=True), skipped


def holm_adjust_pvalues(p_values: List[float]) -> np.ndarray:
    """Apply Holm correction to a list of p-values."""
    if not p_values:
        return np.array([])

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)

    running_max = 0.0
    n = len(values)
    for rank, idx in enumerate(order):
        current = min((n - rank) * values[idx], 1.0)
        running_max = max(running_max, current)
        adjusted[idx] = running_max

    return adjusted


def kruskal_wallis_test(df: pd.DataFrame, agent_type: str, cluster_name: str, conditions: List[str]) -> Tuple[float, float]:
    """Perform Kruskal-Wallis test across conditions within an agent type and cluster."""
    pct_col = f"{cluster_name}_pct"
    agent_df = df[df["agent_type"] == agent_type]
    
    groups = []
    for condition in conditions:
        data = agent_df[agent_df["allies_condition"] == condition][pct_col].dropna().values
        if len(data) > 0:
            groups.append(data)
    
    if len(groups) < 2:
        return np.nan, np.nan
    
    h_stat, p_val = kruskal(*groups)
    return h_stat, p_val


def perform_statistical_tests(df: pd.DataFrame) -> List[StatResult]:
    """Run Kruskal-Wallis and Mann-Whitney U tests across three ally conditions with Holm correction."""
    results: List[StatResult] = []

    for agent_type in AGENT_TYPES:
        agent_df = df[df["agent_type"] == agent_type]

        for cluster_name in CLUSTER_NAMES:
            pct_col = f"{cluster_name}_pct"

            # Kruskal-Wallis test across all three ally conditions
            h_stat, p_kw = kruskal_wallis_test(df, agent_type, cluster_name, ALLIES_CONDITIONS)

            if not np.isnan(p_kw):
                results.append(
                    StatResult(
                        cluster=cluster_name,
                        agent_type=agent_type,
                        comparison="Kruskal-Wallis across 3 conditions",
                        statistic=h_stat,
                        p_value_raw=p_kw,
                        p_value_holm=p_kw,
                        significant=(p_kw < 0.05),
                        n_standard=0,
                        n_strong=0,
                    )
                )

                # Post-hoc pairwise Mann-Whitney U tests if Kruskal-Wallis is significant
                if p_kw < 0.05:
                    pairwise_comparisons = [
                        ("Standard Allies", "Strong Allies"),
                        ("Standard Allies", "Very Strong Allies"),
                        ("Strong Allies", "Very Strong Allies"),
                    ]
                    p_values = []
                    u_stats = []

                    for cond1, cond2 in pairwise_comparisons:
                        data1 = agent_df[agent_df["allies_condition"] == cond1][pct_col].dropna().values
                        data2 = agent_df[agent_df["allies_condition"] == cond2][pct_col].dropna().values

                        if len(data1) > 0 and len(data2) > 0:
                            u_stat, p_raw = mannwhitneyu(data1, data2, alternative="two-sided")
                            p_values.append(p_raw)
                            u_stats.append(u_stat)
                        else:
                            p_values.append(np.nan)
                            u_stats.append(np.nan)

                    # Apply Holm correction
                    adjusted_p = holm_adjust_pvalues([p for p in p_values if not np.isnan(p)])
                    adj_idx = 0

                    for (cond1, cond2), u_stat, p_raw in zip(pairwise_comparisons, u_stats, p_values):
                        if not np.isnan(p_raw):
                            p_holm = adjusted_p[adj_idx]
                            adj_idx += 1
                        else:
                            p_holm = np.nan

                        n1 = len(agent_df[agent_df["allies_condition"] == cond1][pct_col].dropna().values)
                        n2 = len(agent_df[agent_df["allies_condition"] == cond2][pct_col].dropna().values)

                        results.append(
                            StatResult(
                                cluster=cluster_name,
                                agent_type=agent_type,
                                comparison=f"{cond1} vs {cond2}",
                                statistic=u_stat if not np.isnan(u_stat) else 0.0,
                                p_value_raw=p_raw if not np.isnan(p_raw) else 1.0,
                                p_value_holm=p_holm if not np.isnan(p_holm) else 1.0,
                                significant=(p_holm < 0.05) if not np.isnan(p_holm) else False,
                                n_standard=n1,
                                n_strong=n2,
                            )
                        )

    return results


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and standard deviation by agent type and allies condition."""
    rows = []
    for agent_type in AGENT_TYPES:
        for allies_condition in ALLIES_CONDITIONS:
            subset = df[(df["agent_type"] == agent_type) & (df["allies_condition"] == allies_condition)]
            rows.append(
                {
                    "agent_type": agent_type,
                    "allies_condition": allies_condition,
                    "n_runs": len(subset),
                    "Affiliative_mean": subset["Affiliative_pct"].mean(),
                    "Affiliative_std": subset["Affiliative_pct"].std(),
                    "Internalizing_mean": subset["Internalizing_pct"].mean(),
                    "Internalizing_std": subset["Internalizing_pct"].std(),
                    "Externalizing_mean": subset["Externalizing_pct"].mean(),
                    "Externalizing_std": subset["Externalizing_pct"].std(),
                }
            )
    return pd.DataFrame(rows)


def plot_grouped_bars(df: pd.DataFrame, output_path: str) -> None:
    """Create the supplemental grouped bar chart with three-tier x-axis labels."""
    fig, ax = plt.subplots(figsize=(18, 7))

    category_order = [
        ("Standard Allies", "DB"),
        ("Standard Allies", "NB"),
        ("Strong Allies", "DB"),
        ("Strong Allies", "NB"),
        ("Very Strong Allies", "DB"),
        ("Very Strong Allies", "NB"),
    ]
    category_labels = [agent_type for _, agent_type in category_order]
    top_ticks = [0.5, 2.5, 4.5]
    top_labels = ["Standard Allies", "Strong Allies", "Very Strong Allies"]

    x_pos = np.arange(len(category_order))
    bar_width = 0.24
    cluster_colors = {
        "Affiliative": "#2ecc71",
        "Internalizing": "#e74c3c",
        "Externalizing": "#3498db",
    }

    for cluster_idx, cluster_name in enumerate(CLUSTER_NAMES):
        pct_col = f"{cluster_name}_pct"
        means = []
        sems = []

        for allies_condition, agent_type in category_order:
            subset = df[(df["allies_condition"] == allies_condition) & (df["agent_type"] == agent_type)][pct_col].dropna().values
            if len(subset) > 0:
                means.append(float(np.mean(subset)))
                sems.append(float(sem(subset)))
            else:
                means.append(0.0)
                sems.append(0.0)

        x_offset = x_pos + (cluster_idx - 1) * bar_width
        ax.bar(
            x_offset,
            means,
            bar_width,
            label=cluster_name,
            color=cluster_colors[cluster_name],
            alpha=0.85,
            edgecolor="black",
            linewidth=1.0,
            yerr=sems,
            capsize=5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(category_labels, fontsize=10)
    ax.set_xlabel("Agent Type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Cluster Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, 5.6)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    top_ax = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    top_ax.set_xticks(top_ticks)
    top_ax.set_xticklabels(top_labels, fontsize=11, fontweight="bold")
    top_ax.set_xlabel("Allies Condition", fontsize=12, fontweight="bold", labelpad=10)
    top_ax.tick_params(axis="x", length=0, pad=10)

    fig.suptitle(
        "Supplemental Figure: Behavioral Cluster Composition across Allies Conditions",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplemental figure: Allies conditions comparison")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_LowThreat*.csv"),
            os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_NB_LowThreat*.csv"),
            os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_StrongAllies_LowThreat*.csv"),
            os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_NB_StrongAllies_LowThreat*.csv"),
            os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_VeryStrongAllies_LowThreat*.csv"),
            os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_NB_VeryStrongAllies_LowThreat*.csv"),
        ],
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

    summary_df = build_summary_table(raw_df)
    stat_results = perform_statistical_tests(raw_df)

    fig_path = os.path.join(args.output_dir, "SuppStrongAllies_grouped_bars.png")
    raw_path = os.path.join(args.output_dir, "SuppStrongAllies_raw_run_percentages.csv")
    stats_path = os.path.join(args.output_dir, "SuppStrongAllies_stats.csv")
    summary_path = os.path.join(args.output_dir, "SuppStrongAllies_summary.csv")
    skipped_path = os.path.join(args.output_dir, "SuppStrongAllies_skipped_files.csv")

    plot_grouped_bars(raw_df, fig_path)

    export_cols = ["agent_type", "allies_condition", "source_file"]
    for cluster_name in CLUSTER_NAMES:
        export_cols.append(f"{cluster_name}_pct")
    raw_df[export_cols].to_csv(raw_path, index=False)
    print(f"Saved raw run percentages to {raw_path}")

    stat_df = pd.DataFrame(
        [
            {
                "cluster": r.cluster,
                "agent_type": r.agent_type,
                "comparison": r.comparison,
                "statistic": r.statistic,
                "p_value_raw": r.p_value_raw,
                "p_value_holm": r.p_value_holm,
                "significant": r.significant,
                "n_standard": r.n_standard,
                "n_strong": r.n_strong,
            }
            for r in stat_results
        ]
    )
    stat_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to {stats_path}")

    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to {summary_path}")

    if skipped:
        pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(skipped_path, index=False)
        print(f"Saved skipped files list to {skipped_path}")

    print("\n" + "=" * 100)
    print("Supplemental Allies Conditions Figure Summary")
    print("=" * 100)
    print(f"Total runs analyzed: {len(raw_df)}")
    print(f"Agent types: {AGENT_TYPES}")
    print(f"Allies conditions: {ALLIES_CONDITIONS}")
    print("\nSummary by Agent Type and Allies Condition:")
    print(summary_df.to_string(index=False))

    if not stat_df.empty:
        print(f"\nStatistical Test Results ({len(stat_df)} tests):")
        print(stat_df.to_string(index=False))


if __name__ == "__main__":
    main()
