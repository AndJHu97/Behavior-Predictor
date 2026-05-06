#!/usr/bin/env python3
"""
Figure 3v2: Agent-type heatmap of adaptive and maladaptive behavior percentages.

This script reads run-level multiple-runs CSV files for DB and NB agents across
LowThreat, ModerateThreat, and HighThreat conditions, computes run-level behavior
percentages, and renders two heatmaps:

- Left: DB agents
- Right: NB agents

Each heatmap shows 11 rows of behaviors and 3 condition columns.
Rows are grouped as:

Adaptive:
- Healthy friendliness
- Protective behavior
- Positive expectancy
- Help-seeking
- Adaptive avoidance

Maladaptive:
- Hypervigilant withdrawal
- Misdirected aggression
- Relational aggression
- Dangerous trust
- Negative expectancy
- Learned helplessness

Positive and negative expectancy use the expectancy denominator
(hopefulness + cynical + learned_helplessness). All other displayed behaviors
use the selected complex-behavior denominator (all displayed complex-behavior
rows excluding expectancies).

Analysis exported to CSV:
- Heatmap summary table (mean/std/sem by agent, condition, behavior)
- Primary MANOVA: adaptive and maladaptive clusters across conditions within each agent
- Secondary MANOVA: adaptive and maladaptive clusters comparing DB vs NB within each condition
- Raw run-level percentages
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

try:
    from statsmodels.multivariate.manova import MANOVA

    HAS_MANOVA = True
except ImportError:  # pragma: no cover
    HAS_MANOVA = False


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "Figures")
OUTPUT_DIR = FIGURES_DIR

CONDITIONS = ["No ACE", "Moderate adversity", "ACE"]
CONDITION_PATTERNS = {
    "No ACE": ["LowThreat"],
    "Moderate adversity": ["ModerateThreat"],
    "ACE": ["HighThreat"],
}
AGENT_TYPES = ["DB", "NB"]

ROW_SPECS = [
    ("healthy_friendliness", "Healthy friendliness", "adaptive", "complex"),
    ("protective_behavior", "Protective behavior", "adaptive", "complex"),
    ("hopefulness", "Positive expectancy", "adaptive", "expectancy"),
    ("community_trusting_vulnerability", "Help-seeking", "adaptive", "complex"),
    ("willingness_to_flee", "Adaptive avoidance", "adaptive", "complex"),
    ("fearful_withdrawn_relationship", "Hypervigilant withdrawal", "maladaptive", "complex"),
    ("bully_behavior", "Misdirected aggression", "maladaptive", "complex"),
    ("aggressive_withdrawn_relationship", "Relational aggression", "maladaptive", "complex"),
    ("dangerous_trust", "Dangerous trust", "maladaptive", "complex"),
    ("cynical", "Negative expectancy", "maladaptive", "expectancy"),
    ("learned_helplessness", "Learned helplessness", "maladaptive", "complex"),
]

ADAPTIVE_KEYS = [
    "healthy_friendliness",
    "protective_behavior",
    "hopefulness",
    "community_trusting_vulnerability",
    "willingness_to_flee",
]

MALADAPTIVE_KEYS = [
    "fearful_withdrawn_relationship",
    "bully_behavior",
    "aggressive_withdrawn_relationship",
    "dangerous_trust",
    "cynical",
    "learned_helplessness",
]

ROW_ORDER = [spec[0] for spec in ROW_SPECS]
ROW_LABELS = [spec[1] for spec in ROW_SPECS]

COLUMN_ALIASES = {
    "protective_behavior": ["protective_behavior"],
    "community_trusting_vulnerability": ["community_trusting_vulnerability"],
    "healthy_friendliness": ["healthy_friendliness"],
    "bully_behavior": ["bully_behavior"],
    "aggressive_withdrawn_relationship": ["aggressive_withdrawn_relationship"],
    "fearful_withdrawn_relationship": ["fearful_withdrawn_relationship"],
    "learned_helplessness": ["learned_helplessness", "Learned_helplessness"],
    "dangerous_trust": ["dangerous_trust"],
    "willingness_to_flee": ["willingness_to_flee"],
    "hopefulness": ["hopefulness"],
    "cynical": ["cynical"],
}


@dataclass
class ManovaRow:
    analysis_type: str
    cluster: str
    comparison: str
    group_label: str
    predictor: str
    dependent_variables: str
    statistic: str
    value: float
    num_df: float
    den_df: float
    f_value: float
    p_value: float


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def flatten_globs(patterns: Iterable[str], include_series: bool = False) -> List[str]:
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
    name = os.path.basename(path).lower()
    if "_db_" in name or name.startswith("db_") or "multiple_runs_db" in name:
        return "DB"
    if "_nb_" in name or name.startswith("nb_") or "multiple_runs_nb" in name:
        return "NB"
    return None


def infer_condition(path: str) -> Optional[str]:
    name = os.path.basename(path)
    for condition, patterns in CONDITION_PATTERNS.items():
        if any(pattern.lower() in name.lower() for pattern in patterns):
            return condition
    return None


def find_run_header(rows: List[List[str]]) -> int:
    for index, row in enumerate(rows):
        if row and row[0].strip().lower() == "run":
            return index
    return -1


def resolve_columns(header: List[str], needed_cols: List[str]) -> Dict[str, int]:
    normalized = {col.strip().lower(): idx for idx, col in enumerate(header)}
    if "run" not in normalized:
        raise ValueError("Run column missing")

    index_map = {"Run": normalized["run"]}
    for col in needed_cols:
        aliases = COLUMN_ALIASES[col]
        found = None
        for alias in aliases:
            key = alias.lower()
            if key in normalized:
                found = normalized[key]
                break
        if found is None:
            raise ValueError(f"Missing required column: {col}")
        index_map[col] = found
    return index_map


def parse_run_rows(csv_path: str, needed_cols: List[str]) -> List[Tuple[int, Dict[str, float]]]:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    run_header_idx = find_run_header(rows)
    if run_header_idx < 0:
        return []

    header = [col.strip() for col in rows[run_header_idx]]
    index_map = resolve_columns(header, needed_cols)

    parsed: List[Tuple[int, Dict[str, float]]] = []
    for row in rows[run_header_idx + 1 :]:
        if not row:
            break
        first = row[0].strip().lower()
        if first in {"statistic", "total", "mean", "std", "min", "max"}:
            break
        try:
            run_id = int(float(row[index_map["Run"]]))
            values = {col: float(row[index_map[col]]) for col in needed_cols}
        except (ValueError, IndexError):
            continue
        parsed.append((run_id, values))

    return parsed


def safe_percent(numerator: float, denominator: float) -> float:
    return (numerator / denominator) * 100.0 if denominator else 0.0


def load_all_runs(files: List[str]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    complex_cols = [
        "healthy_friendliness",
        "protective_behavior",
        "community_trusting_vulnerability",
        "willingness_to_flee",
        "fearful_withdrawn_relationship",
        "bully_behavior",
        "aggressive_withdrawn_relationship",
        "dangerous_trust",
        "learned_helplessness",
        "hopefulness",
        "cynical",
    ]

    rows: List[Dict[str, float]] = []
    skipped: List[Tuple[str, str]] = []

    for path in files:
        agent_type = infer_agent_type(path)
        condition = infer_condition(path)
        if agent_type is None or condition is None:
            skipped.append((path, "Could not infer agent type or condition from filename"))
            continue

        try:
            run_rows = parse_run_rows(path, complex_cols)
            if not run_rows:
                skipped.append((path, "Missing or invalid Run table for required behavior columns"))
                continue

            for run_id, values in run_rows:
                selected_complex_total = sum(
                    values[col]
                    for col in [
                        "healthy_friendliness",
                        "protective_behavior",
                        "community_trusting_vulnerability",
                        "willingness_to_flee",
                        "fearful_withdrawn_relationship",
                        "bully_behavior",
                        "aggressive_withdrawn_relationship",
                        "dangerous_trust",
                        "learned_helplessness",
                    ]
                )
                total_expectancy = values["hopefulness"] + values["cynical"] + values["learned_helplessness"]

                row = {
                    "source_file": os.path.basename(path),
                    "source_path": path,
                    "agent_type": agent_type,
                    "condition": condition,
                    "run": run_id,
                    "selected_complex_total": selected_complex_total,
                    "total_expectancy": total_expectancy,
                }

                for key in ROW_ORDER:
                    if key == "hopefulness":
                        row[key] = safe_percent(values["hopefulness"], total_expectancy)
                        row[f"{key}_count"] = values["hopefulness"]
                        row[f"{key}_denominator"] = total_expectancy
                    elif key == "cynical":
                        cynical_count = values["cynical"] + values["learned_helplessness"]
                        row[key] = safe_percent(cynical_count, total_expectancy)
                        row[f"{key}_count"] = cynical_count
                        row[f"{key}_denominator"] = total_expectancy
                    else:
                        row[key] = safe_percent(values[key], selected_complex_total)
                        row[f"{key}_count"] = values[key]
                        row[f"{key}_denominator"] = selected_complex_total

                rows.append(row)
        except Exception as exc:
            skipped.append((path, str(exc)))

    if not rows:
        return pd.DataFrame(), skipped
    return pd.DataFrame(rows), skipped


def build_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent_type in AGENT_TYPES:
        for condition in CONDITIONS:
            subset = raw_df[(raw_df["agent_type"] == agent_type) & (raw_df["condition"] == condition)]
            for key, label, category, denominator_type in ROW_SPECS:
                values = subset[key].to_numpy(dtype=float)
                if values.size == 0:
                    continue
                rows.append(
                    {
                        "agent_type": agent_type,
                        "condition": condition,
                        "behavior_key": key,
                        "behavior_label": label,
                        "category": category,
                        "denominator_type": denominator_type,
                        "n_runs": int(values.size),
                        "mean_pct": float(np.mean(values)),
                        "std_pct": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                        "sem_pct": float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
                        "min_pct": float(np.min(values)),
                        "max_pct": float(np.max(values)),
                    }
                )
    return pd.DataFrame(rows)


def _resolve_manova_result_key(mv_results, predictor: str) -> Optional[str]:
    candidate_keys = [predictor, f"C({predictor})", f"Q('{predictor}')", f'Q("{predictor}")']
    for candidate in candidate_keys:
        if candidate in mv_results.results:
            return candidate
    for candidate in mv_results.results.keys():
        if predictor in candidate:
            return candidate
    return None


def _extract_manova_rows(
    mv_results,
    analysis_type: str,
    cluster: str,
    comparison: str,
    group_label: str,
    predictor: str,
    dependent_variables: str,
) -> List[ManovaRow]:
    rows: List[ManovaRow] = []
    result_key = _resolve_manova_result_key(mv_results, predictor)
    if result_key is None:
        return rows

    try:
        stat_df = mv_results.results[result_key]["stat"]
    except Exception:
        return rows

    for stat_name, stat_row in stat_df.iterrows():
        rows.append(
            ManovaRow(
                analysis_type=analysis_type,
                cluster=cluster,
                comparison=comparison,
                group_label=group_label,
                predictor=predictor,
                dependent_variables=dependent_variables,
                statistic=str(stat_name),
                value=float(stat_row.get("Value", np.nan)),
                num_df=float(stat_row.get("Num DF", np.nan)),
                den_df=float(stat_row.get("Den DF", np.nan)),
                f_value=float(stat_row.get("F Value", np.nan)),
                p_value=float(stat_row.get("Pr > F", np.nan)),
            )
        )
    return rows


def filter_manova_keys(subset: pd.DataFrame, keys: List[str]) -> Tuple[List[str], List[str]]:
    usable_keys: List[str] = []
    dropped_keys: List[str] = []
    for key in keys:
        series = subset[key].dropna()
        if series.nunique() > 1 and float(series.std(ddof=0)) > 0.0:
            usable_keys.append(key)
        else:
            dropped_keys.append(key)
    return usable_keys, dropped_keys


def run_manova_analysis(raw_df: pd.DataFrame) -> pd.DataFrame:
    if not HAS_MANOVA:
        return pd.DataFrame(
            [
                {
                    "analysis_type": "error",
                    "cluster": "all",
                    "comparison": "all",
                    "group_label": "all",
                    "predictor": "statsmodels_unavailable",
                    "statistic": "error",
                    "value": np.nan,
                    "num_df": np.nan,
                    "den_df": np.nan,
                    "f_value": np.nan,
                    "p_value": np.nan,
                    "error": "statsmodels MANOVA unavailable",
                }
            ]
        )

    analysis_rows: List[ManovaRow] = []

    cluster_map = {
        "adaptive": ADAPTIVE_KEYS,
        "maladaptive": MALADAPTIVE_KEYS,
    }

    # Primary analysis: within each agent type, conditions differ.
    for agent_type in AGENT_TYPES:
        subset = raw_df[raw_df["agent_type"] == agent_type].copy()
        if subset.empty:
            continue

        for cluster_name, keys in cluster_map.items():
            usable_keys, dropped_keys = filter_manova_keys(subset, keys)
            dependent = " + ".join(usable_keys)
            if len(usable_keys) < 2:
                analysis_rows.append(
                    ManovaRow(
                        analysis_type="within_agent_conditions",
                        cluster=cluster_name,
                        comparison=agent_type,
                        group_label="condition",
                        predictor="condition",
                        dependent_variables="|".join(usable_keys),
                        statistic="insufficient_variability",
                        value=np.nan,
                        num_df=np.nan,
                        den_df=np.nan,
                        f_value=np.nan,
                        p_value=np.nan,
                    )
                )
                continue

            formula = f"{dependent} ~ C(condition)"
            try:
                mv = MANOVA.from_formula(formula, data=subset)
                res = mv.mv_test()
                analysis_rows.extend(
                    _extract_manova_rows(
                        res,
                        analysis_type="within_agent_conditions",
                        cluster=cluster_name,
                        comparison=agent_type,
                        group_label="condition",
                        predictor="condition",
                        dependent_variables="|".join(usable_keys),
                    )
                )
            except Exception:
                analysis_rows.append(
                    ManovaRow(
                        analysis_type="within_agent_conditions",
                        cluster=cluster_name,
                        comparison=agent_type,
                        group_label="condition",
                        predictor="condition",
                        dependent_variables="|".join(usable_keys),
                        statistic="error",
                        value=np.nan,
                        num_df=np.nan,
                        den_df=np.nan,
                        f_value=np.nan,
                        p_value=np.nan,
                    )
                )

    # Secondary analysis: within each condition, agent types differ.
    for condition in CONDITIONS:
        subset = raw_df[raw_df["condition"] == condition].copy()
        if subset.empty:
            continue

        for cluster_name, keys in cluster_map.items():
            usable_keys, dropped_keys = filter_manova_keys(subset, keys)
            dependent = " + ".join(usable_keys)
            if len(usable_keys) < 2:
                analysis_rows.append(
                    ManovaRow(
                        analysis_type="within_condition_agents",
                        cluster=cluster_name,
                        comparison=condition,
                        group_label="agent_type",
                        predictor="agent_type",
                        dependent_variables="|".join(usable_keys),
                        statistic="insufficient_variability",
                        value=np.nan,
                        num_df=np.nan,
                        den_df=np.nan,
                        f_value=np.nan,
                        p_value=np.nan,
                    )
                )
                continue

            formula = f"{dependent} ~ C(agent_type)"
            try:
                mv = MANOVA.from_formula(formula, data=subset)
                res = mv.mv_test()
                analysis_rows.extend(
                    _extract_manova_rows(
                        res,
                        analysis_type="within_condition_agents",
                        cluster=cluster_name,
                        comparison=condition,
                        group_label="agent_type",
                        predictor="agent_type",
                        dependent_variables="|".join(usable_keys),
                    )
                )
            except Exception:
                analysis_rows.append(
                    ManovaRow(
                        analysis_type="within_condition_agents",
                        cluster=cluster_name,
                        comparison=condition,
                        group_label="agent_type",
                        predictor="agent_type",
                        dependent_variables="|".join(usable_keys),
                        statistic="error",
                        value=np.nan,
                        num_df=np.nan,
                        den_df=np.nan,
                        f_value=np.nan,
                        p_value=np.nan,
                    )
                )

    return pd.DataFrame([row.__dict__ for row in analysis_rows])


def plot_heatmaps(summary_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 13), sharey=True, constrained_layout=True)
    vmax = 100.0
    vmin = 0.0
    cmap = plt.get_cmap("RdBu_r")

    for axis, agent_type in zip(axes, AGENT_TYPES):
        matrix = np.zeros((len(ROW_ORDER), len(CONDITIONS)), dtype=float)
        annotation = np.empty_like(matrix, dtype=object)

        for row_idx, row_key in enumerate(ROW_ORDER):
            for col_idx, condition in enumerate(CONDITIONS):
                match = summary_df[
                    (summary_df["agent_type"] == agent_type)
                    & (summary_df["condition"] == condition)
                    & (summary_df["behavior_key"] == row_key)
                ]
                if match.empty:
                    value = np.nan
                else:
                    value = float(match.iloc[0]["mean_pct"])
                matrix[row_idx, col_idx] = value if np.isfinite(value) else 0.0
                annotation[row_idx, col_idx] = "" if not np.isfinite(value) else f"{value:.1f}"

        # Use a diverging normalization centered at 0 so that 0 maps to the
        # center color (white) of the diverging colormap.
        # TwoSlopeNorm requires vmin < vcenter < vmax; ensure a tiny negative
        # lower bound when vmin would equal the center (0).
        vmin_norm = vmin
        if vmin_norm >= 0.0:
            vmin_norm = -1e-6 * (vmax if vmax > 0 else 1.0)
        norm = TwoSlopeNorm(vmin=vmin_norm, vcenter=0.0, vmax=vmax)
        im = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        # Force the displayed color limits to start at 0 so the colorbar shows
        # 0..vmax and no negative labels (the tiny negative vmin is only used
        # to satisfy TwoSlopeNorm's requirement).
        try:
            im.set_clim(0.0, vmax)
        except Exception:
            pass
        axis.set_xticks(np.arange(len(CONDITIONS)))
        axis.set_xticklabels(CONDITIONS, rotation=0, fontsize=11)
        axis.set_yticks(np.arange(len(ROW_LABELS)))
        axis.set_yticklabels(ROW_LABELS, fontsize=9)
        axis.set_title(f"{agent_type} agents", fontsize=14, fontweight="bold", pad=15)

        # Group separators.
        axis.axhline(4.5, color="white", linewidth=2.0, alpha=0.95)
        axis.axhline(9.5, color="black", linewidth=3.0, alpha=0.95)

        # Annotate cells.
        threshold = 0.55 * vmax
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                text = annotation[row_idx, col_idx]
                if not text:
                    continue
                color = "white" if matrix[row_idx, col_idx] >= threshold else "black"
                axis.text(col_idx, row_idx, text, ha="center", va="center", fontsize=9, color=color)

        axis.set_xlim(-0.5, len(CONDITIONS) - 0.5)
        axis.set_ylim(len(ROW_LABELS) - 0.5, -0.5)
        axis.grid(False)
        
        # Adjust margins to prevent label cutoff
        axis.margins(x=0.01, y=0.01)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Mean percentage", fontsize=12)
    # Force colorbar ticks and labels to the 0..vmax range so negative tiny
    # values used internally are not shown to the user.
    try:
        ticks = np.linspace(0.0, vmax, 6)
        im.set_clim(0.0, vmax)
        cbar.update_normal(im)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
    except Exception:
        pass

    fig.suptitle("Figure 3v2: Behavioral heatmap by agent type and adversity condition", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv_with_fallback(df: pd.DataFrame, target_path: str) -> str:
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        root, ext = os.path.splitext(target_path)
        alt_path = f"{root}_updated{ext}"
        df.to_csv(alt_path, index=False)
        return alt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 3v2 heatmap and MANOVA analysis")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[os.path.join(WORKSPACE_ROOT, "**", "multiple_runs*Threat*.csv")],
        help="Input CSV files or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--include-series",
        action="store_true",
        help="Include CSVs with 'series' in their filename (disable default exclusion)",
    )
    args = parser.parse_args()

    ensure_output_dir()
    files = flatten_globs(args.inputs, include_series=args.include_series)
    if not files:
        raise SystemExit("No matching run CSV files found.")

    raw_df, skipped = load_all_runs(files)
    if raw_df.empty:
        raise SystemExit("No valid run-level rows parsed from the input files.")

    summary_df = build_summary_table(raw_df)
    manova_df = run_manova_analysis(raw_df)

    fig_path = os.path.join(args.output_dir, "figure3v2_heatmap.png")
    raw_path = os.path.join(args.output_dir, "figure3v2_raw_run_percentages.csv")
    summary_path = os.path.join(args.output_dir, "figure3v2_heatmap_summary.csv")
    manova_path = os.path.join(args.output_dir, "figure3v2_manova_results.csv")
    skipped_path = os.path.join(args.output_dir, "figure3v2_skipped_files.csv")
    report_path = os.path.join(args.output_dir, "figure3v2_analysis_summary.txt")

    plot_heatmaps(summary_df, fig_path)

    raw_saved = write_csv_with_fallback(raw_df, raw_path)
    summary_saved = write_csv_with_fallback(summary_df, summary_path)
    manova_saved = write_csv_with_fallback(manova_df, manova_path)
    skipped_df = pd.DataFrame(skipped, columns=["source_path", "reason"]) if skipped else pd.DataFrame(columns=["source_path", "reason"])
    skipped_saved = write_csv_with_fallback(skipped_df, skipped_path)

    lines = [
        "Figure 3v2 heatmap analysis summary",
        "=" * 72,
        f"Parsed run files: {len(files)}",
        f"Valid runs analyzed: {len(raw_df)}",
        f"DB runs: {int((raw_df['agent_type'] == 'DB').sum())}",
        f"NB runs: {int((raw_df['agent_type'] == 'NB').sum())}",
        "",
        "Outputs:",
        f"  Figure: {fig_path}",
        f"  Raw run table: {raw_saved}",
        f"  Heatmap summary: {summary_saved}",
        f"  MANOVA results: {manova_saved}",
        f"  Skipped-file log: {skipped_saved}",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved figure: {fig_path}")
    print(f"Saved raw run table: {raw_saved}")
    print(f"Saved heatmap summary: {summary_saved}")
    print(f"Saved MANOVA results: {manova_saved}")
    print(f"Saved skipped-file log: {skipped_saved}")
    print(f"Saved analysis summary: {report_path}")


if __name__ == "__main__":
    main()
