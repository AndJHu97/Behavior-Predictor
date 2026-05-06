#!/usr/bin/env python3
"""
Supplemental Figure: Strong Allies vs Standard Allies Heatmap (DB Agents Only)

This script creates a heatmap comparing individual complex behaviors across
Standard Allies and Strong Allies conditions for DB agents only.

Behaviors are organized into three clusters:
- Affiliative: healthy_friendliness, community_trusting_vulnerability, protective_behavior
- Internalizing: fearful_withdrawn_relationship, willingness_to_flee, learned_helplessness
- Externalizing: bully_behavior, aggressive_withdrawn_relationship, dangerous_trust

Heatmap design:
- Rows: Standard Allies, Strong Allies (conditions)
- Columns: Individual behaviors with two-tier labels (cluster grouping | behavior name)
- Color: White (0%) to Dark Red (100%) - dark red from figure3v2 RdBu_r colormap (#B2182B)

Statistical analysis:
- Mann-Whitney U test for each behavior comparing Standard Allies vs Strong Allies.
- Holm correction applied across all 9 behaviors.
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, sem

try:
    from statsmodels.stats.multitest import multipletests
    HAS_MULTIPLETESTS = True
except ImportError:
    HAS_MULTIPLETESTS = False


# Behavior-to-cluster mappings.
AFFILIATIVE = ["healthy_friendliness", "community_trusting_vulnerability", "protective_behavior"]
INTERNALIZING = ["fearful_withdrawn_relationship", "willingness_to_flee", "learned_helplessness"]
EXTERNALIZING = ["bully_behavior", "aggressive_withdrawn_relationship", "dangerous_trust"]

ALL_BEHAVIORS = AFFILIATIVE + INTERNALIZING + EXTERNALIZING
CLUSTER_NAMES = ["Affiliative", "Internalizing", "Externalizing"]
CLUSTER_BEHAVIORS = [AFFILIATIVE, INTERNALIZING, EXTERNALIZING]

# Human-readable behavior names for display
BEHAVIOR_DISPLAY_NAMES = {
    "healthy_friendliness": "Healthy Friendliness",
    "community_trusting_vulnerability": "Help-Seeking Vulnerability",
    "protective_behavior": "Protective Behavior",
    "fearful_withdrawn_relationship": "Hypervigilant Withdrawal",
    "willingness_to_flee": "Adaptive Avoidance",
    "learned_helplessness": "Learned Helplessness",
    "bully_behavior": "Misdirected Aggression",
    "aggressive_withdrawn_relationship": "Relational Aggression",
    "dangerous_trust": "Dangerous Trust",
}

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "Figures")
OUTPUT_DIR = FIGURES_DIR


@dataclass
class StatResult:
    behavior: str
    cluster: str
    comparison: str
    statistic: float
    p_value_raw: float
    p_value_holm: float
    significant: bool
    n_standard: int
    n_strong: int


@dataclass
class AnalysisSpec:
    name: str
    slug: str
    analysis_key: str
    patterns: List[str]
    condition_order: List[str]
    standard_condition: str
    strong_condition: str
    ace_condition: str


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


def infer_condition_label(path: str, analysis: str) -> Optional[str]:
    """Infer the appropriate condition label for the selected analysis."""
    lower = path.lower()
    has_strong = "strongallies" in lower
    is_low = "lowthreat" in lower
    is_moderate = "moderatethreat" in lower
    is_high = "highthreat" in lower

    if analysis == "low":
        if is_low and has_strong:
            return "Strong Allies (Low Threat)"
        if is_low and not has_strong:
            return "Standard Allies (Low Threat)"
        if is_high and not has_strong:
            return "ACE Standard Allies"
    elif analysis == "moderate":
        if is_moderate and has_strong:
            return "Strong Allies (Moderate Threat)"
        if is_moderate and not has_strong:
            return "Standard Allies (Moderate Threat)"
        if is_high and not has_strong:
            return "ACE Standard Allies"
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
        row = {}
        for col, idx in col_indices.items():
            try:
                row[col] = float(cells[idx]) if idx < len(cells) else 0.0
            except (ValueError, IndexError):
                row[col] = 0.0
        rows_data.append(row)

    return pd.DataFrame(rows_data) if rows_data else pd.DataFrame()


def load_all_runs(files: List[str], analysis: str) -> pd.DataFrame:
    """Load all runs for one analysis, parsing and computing behavior percentages."""
    all_data = []

    for file_path in files:
        allies_condition = infer_condition_label(file_path, analysis)
        if allies_condition is None:
            continue

        df = parse_run_rows(file_path, ALL_BEHAVIORS)
        if df.empty:
            continue

        for _, row in df.iterrows():
            total = sum(row[b] for b in ALL_BEHAVIORS if b in row)
            if total == 0:
                total = 1.0

            run_data = {
                "file": file_path,
                "allies_condition": allies_condition,
            }

            for behavior in ALL_BEHAVIORS:
                pct = 100.0 * row.get(behavior, 0.0) / total
                run_data[behavior] = pct

            all_data.append(run_data)

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def holm_adjust_pvalues(p_values: List[float]) -> List[float]:
    """Manual implementation of Holm step-down correction."""
    if not p_values:
        return []

    n = len(p_values)
    indices = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n

    for rank, i in enumerate(indices):
        adjusted[i] = p_values[i] * (n - rank)
        if rank > 0:
            adjusted[i] = max(adjusted[i], adjusted[indices[rank - 1]])
        adjusted[i] = min(adjusted[i], 1.0)

    return adjusted


def perform_statistical_tests(df: pd.DataFrame, conditions: List[str]) -> Tuple[List[StatResult], List[Dict[str, object]]]:
    """Perform Kruskal-Wallis across all conditions and pairwise Mann-Whitney U tests with Holm correction.

    Returns a list of StatResult for pairwise comparisons; Kruskal-Wallis results are saved separately
    in a dict returned as the second element.
    """
    pairwise_results = []
    pairwise_pvals = []
    pairwise_meta = []

    kw_results = []

    for behavior in ALL_BEHAVIORS:
        # Only analyze behaviors within their cluster (maintain previous behavior->cluster mapping)
        cluster_name = next((cn for cn, blist in zip(CLUSTER_NAMES, CLUSTER_BEHAVIORS) if behavior in blist), "")

        # Gather data per condition
        groups = [df[df["allies_condition"] == c][behavior].dropna() for c in conditions]
        counts = [len(g) for g in groups]

        # Kruskal-Wallis if at least two groups have data
        nonempty = sum(1 for c in groups if len(c) > 0)
        if nonempty >= 2:
            try:
                from scipy.stats import kruskal

                kw_stat, kw_p = kruskal(*[g for g in groups if len(g) > 0])
            except Exception:
                kw_stat, kw_p = (np.nan, np.nan)
        else:
            kw_stat, kw_p = (np.nan, np.nan)

        kw_results.append({
            "behavior": behavior,
            "cluster": cluster_name,
            "kw_stat": kw_stat,
            "kw_p": kw_p,
            "n_standard": counts[0],
            "n_strong": counts[1],
            "n_ace": counts[2],
        })

        # Pairwise comparisons (all unique pairs)
        pairs = [ (0,1), (0,2), (1,2) ]
        for i,j in pairs:
            gi = groups[i]
            gj = groups[j]
            if len(gi) > 0 and len(gj) > 0:
                stat, p_raw = mannwhitneyu(gi, gj, alternative="two-sided")
                pairwise_pvals.append(p_raw)
                pairwise_meta.append((behavior, cluster_name, i, j, stat, len(gi), len(gj)))

    # Holm adjust across all pairwise p-values
    p_adj = holm_adjust_pvalues(pairwise_pvals)

    for idx, meta in enumerate(pairwise_meta):
        behavior, cluster_name, i, j, stat, ni, nj = meta
        pa = p_adj[idx]
        comp = f"{conditions[i]} vs {conditions[j]}"
        pairwise_results.append(StatResult(
            behavior=behavior,
            cluster=cluster_name,
            comparison=comp,
            statistic=stat,
            p_value_raw=pairwise_pvals[idx],
            p_value_holm=pa,
            significant=pa < 0.05,
            n_standard=ni,
            n_strong=nj,
        ))

    return pairwise_results, kw_results


def build_heatmap_data(df: pd.DataFrame, conditions: List[str]) -> pd.DataFrame:
    """Build heatmap data with behaviors as rows and conditions as columns.

    Columns order follows the provided conditions.
    """
    rows = []
    for behavior in ALL_BEHAVIORS:
        vals = {}
        for cond in conditions:
            pcts = df[df["allies_condition"] == cond][behavior]
            vals[cond] = pcts.mean() if len(pcts) > 0 else 0.0

        row = {"behavior": behavior}
        for cond in conditions:
            row[cond] = vals[cond]
        rows.append(row)

    return pd.DataFrame(rows)


def plot_heatmap(
    heatmap_df: pd.DataFrame,
    stat_results: List[StatResult],
    output_path: str,
    condition_cols: List[str],
    title: str,
) -> None:
    """Create a heatmap with behaviors as rows and the provided conditions as columns."""
    fig, ax = plt.subplots(figsize=(11, 8))

    data_matrix = heatmap_df[condition_cols].values
    cmap = LinearSegmentedColormap.from_list("white_darkred", ["white", "#B2182B"])
    im = ax.imshow(data_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(condition_cols)))
    ax.set_xticklabels(condition_cols, fontsize=11, fontweight="bold")

    row_labels = [BEHAVIOR_DISPLAY_NAMES.get(b, b) for b in ALL_BEHAVIORS]
    ax.set_yticks(np.arange(len(ALL_BEHAVIORS)))
    ax.set_yticklabels(row_labels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, label="Percentage (%)", ticks=[0, 25, 50, 75, 100])
    cbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.axhline(3 - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axhline(6 - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    cell_marks = [[0 for _ in condition_cols] for _ in ALL_BEHAVIORS]
    behavior_index = {behavior: idx for idx, behavior in enumerate(ALL_BEHAVIORS)}
    condition_index = {condition: idx for idx, condition in enumerate(condition_cols)}
    for result in stat_results:
        if not result.significant:
            continue
        if " vs " not in result.comparison:
            continue
        left, right = result.comparison.split(" vs ", 1)
        if left not in condition_index or right not in condition_index:
            continue
        row_idx = behavior_index.get(result.behavior)
        if row_idx is None:
            continue
        left_idx = condition_index[left]
        right_idx = condition_index[right]
        left_value = float(data_matrix[row_idx, left_idx])
        right_value = float(data_matrix[row_idx, right_idx])
        mark_idx = left_idx if left_value >= right_value else right_idx
        cell_marks[row_idx][mark_idx] += 1

    for row_idx in range(data_matrix.shape[0]):
        for col_idx in range(data_matrix.shape[1]):
            value = data_matrix[row_idx, col_idx]
            text_color = "white" if value >= 50 else "black"
            text_str = f"{value:.1f}" + ("*" * cell_marks[row_idx][col_idx])
            ax.text(col_idx, row_idx, text_str, ha="center", va="center",
                    color=text_color, fontsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_analysis(spec: AnalysisSpec) -> None:
    """Run one heatmap analysis and write outputs for a single threat context."""
    files = flatten_globs(spec.patterns)
    if not files:
        print(f"Error: No CSV files found for {spec.name}.")
        return

    print(f"\n{spec.name}: found {len(files)} files.")
    for file_path in files:
        print(f"  - {os.path.basename(file_path)}")

    df = load_all_runs(files, spec.analysis_key)
    if df.empty:
        print(f"Error: No run data found for {spec.name}.")
        return

    available_conditions = [condition for condition in spec.condition_order if condition in df["allies_condition"].unique()]
    if len(available_conditions) < 3:
        print(f"Error: Expected 3 conditions for {spec.name}, found {available_conditions}.")
        return

    stat_results_pairwise, kw_results = perform_statistical_tests(df, spec.condition_order)
    heatmap_df = build_heatmap_data(df, spec.condition_order)

    print("\n" + "=" * 100)
    print(f"Supplemental Strong Allies Heatmap Summary ({spec.name})")
    print("=" * 100)
    print(f"Total runs analyzed: {len(df)}")
    print(f"Allies conditions: {sorted(df['allies_condition'].unique())}\n")

    print("Behavior Percentages by Condition:")
    summary_df = df.groupby("allies_condition")[ALL_BEHAVIORS].mean()
    print(summary_df.T.reindex(columns=spec.condition_order).to_string())

    print("\n\nKruskal-Wallis Results (per behavior):")
    kw_df = pd.DataFrame(kw_results)
    print(kw_df.to_string(index=False))

    print("\n\nPairwise Mann-Whitney U Results (Holm-corrected across all pairwise tests):")
    stat_df = pd.DataFrame({
        "behavior": [r.behavior for r in stat_results_pairwise],
        "cluster": [r.cluster for r in stat_results_pairwise],
        "comparison": [r.comparison for r in stat_results_pairwise],
        "statistic": [r.statistic for r in stat_results_pairwise],
        "p_value_raw": [r.p_value_raw for r in stat_results_pairwise],
        "p_value_holm": [r.p_value_holm for r in stat_results_pairwise],
        "significant": [r.significant for r in stat_results_pairwise],
        "n_1": [r.n_standard for r in stat_results_pairwise],
        "n_2": [r.n_strong for r in stat_results_pairwise],
    })
    print(stat_df.to_string(index=False))

    safe_slug = spec.slug
    heatmap_path = os.path.join(OUTPUT_DIR, f"SuppStrongAlliesHeatmap_{safe_slug}_heatmap.png")
    stat_csv_path = os.path.join(OUTPUT_DIR, f"SuppStrongAlliesHeatmap_{safe_slug}_stats.csv")
    kw_csv_path = os.path.join(OUTPUT_DIR, f"SuppStrongAlliesHeatmap_{safe_slug}_kruskal_wallis.csv")
    raw_csv_path = os.path.join(OUTPUT_DIR, f"SuppStrongAlliesHeatmap_{safe_slug}_raw_behavior_percentages.csv")
    summary_csv_path = os.path.join(OUTPUT_DIR, f"SuppStrongAlliesHeatmap_{safe_slug}_summary.csv")

    plot_heatmap(
        heatmap_df,
        stat_results_pairwise,
        heatmap_path,
        spec.condition_order,
        f"DB Agents: Behavior Comparison ({spec.name})",
    )
    print(f"\nSaved heatmap visualization to {heatmap_path}")

    stat_df.to_csv(stat_csv_path, index=False)
    print(f"Saved pairwise statistics to {stat_csv_path}")

    kw_df.to_csv(kw_csv_path, index=False)
    print(f"Saved Kruskal-Wallis results to {kw_csv_path}")

    raw_data = []
    for _, row in df.iterrows():
        run_data = {"allies_condition": row["allies_condition"]}
        for behavior in ALL_BEHAVIORS:
            run_data[behavior] = row[behavior]
        raw_data.append(run_data)

    raw_df = pd.DataFrame(raw_data)
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"Saved raw behavior percentages to {raw_csv_path}")

    summary_df.to_csv(summary_csv_path)
    print(f"Saved heatmap summary to {summary_csv_path}")


def main():
    """Main workflow."""
    analyses = [
        AnalysisSpec(
            name="Low Threat",
            slug="low_threat",
            analysis_key="low",
            patterns=[
                os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_LowThreat*.csv"),
                os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_StrongAllies_LowThreat*.csv"),
                os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_HighThreat*.csv"),
            ],
            condition_order=["Standard Allies (Low Threat)", "Strong Allies (Low Threat)", "ACE Standard Allies"],
            standard_condition="Standard Allies (Low Threat)",
            strong_condition="Strong Allies (Low Threat)",
            ace_condition="ACE Standard Allies",
        ),
        AnalysisSpec(
            name="Moderate Threat",
            slug="moderate_threat",
            analysis_key="moderate",
            patterns=[
                os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_ModerateThreat*.csv"),
                os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_StrongAllies_ModerateThreat*.csv"),
                os.path.join(WORKSPACE_ROOT, "**", "multiple_runs_DB_HighThreat*.csv"),
            ],
            condition_order=["Standard Allies (Moderate Threat)", "Strong Allies (Moderate Threat)", "ACE Standard Allies"],
            standard_condition="Standard Allies (Moderate Threat)",
            strong_condition="Strong Allies (Moderate Threat)",
            ace_condition="ACE Standard Allies",
        ),
    ]

    for spec in analyses:
        run_analysis(spec)


if __name__ == "__main__":
    main()
