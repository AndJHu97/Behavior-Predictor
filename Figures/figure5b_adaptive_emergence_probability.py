#!/usr/bin/env python3
"""
Figure 5B: Probability of Emergence of Adaptive Behaviors

Defines emergence as behavior percent >= global pooled 75th percentile threshold.
Outputs:
- grouped probability plot (% runs present) by condition
- run-level raw table with thresholds and presence flags
- threshold table
- probability summary table with 95% CI
- chi-square association table (condition x presence)
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

CONDITIONS = ["No ACE", "Moderate", "ACE"]
ADAPTIVE_COMPLEX_BEHAVIORS = [
    "protective_behavior",
    "healthy_friendliness",
    "willingness_to_flee",
    "community_trusting_vulnerability",
    "hopefulness",
]

MALADAPTIVE_COMPLEX_BEHAVIORS = [
    "bully_behavior",
    "aggressive_withdrawn_relationship",
    "dangerous_trust",
    "cynical",
    "learned_helplessness",
    "fearful_withdrawn_relationship",
]

TOTAL_COMPLEX_BEHAVIOR_COLUMNS = list(dict.fromkeys(ADAPTIVE_COMPLEX_BEHAVIORS + MALADAPTIVE_COMPLEX_BEHAVIORS))

ADAPTIVE_BEHAVIORS = {
    "protective_behavior": "Protective Behavior",
    "healthy_friendliness": "Healthy Friendliness",
    "willingness_to_flee": "Adaptive Avoidance",
    "community_trusting_vulnerability": "Help-Seeking Vulnerability",
    "hopefulness": "Positive Expectancy",
}

EXPECTANCY_DENOM_COLUMNS = ["hopefulness", "cynical", "learned_helplessness"]
ADAPTIVE_EMERGENCE_SPECS = [
    ("protective_behavior", "Protective Behavior", "protective_behavior", "total_complex_behaviors"),
    ("healthy_friendliness", "Healthy Friendliness", "healthy_friendliness", "total_complex_behaviors"),
    ("willingness_to_flee", "Adaptive Avoidance", "willingness_to_flee", "total_complex_behaviors"),
    ("community_trusting_vulnerability", "Help-Seeking Vulnerability", "community_trusting_vulnerability", "total_complex_behaviors"),
    ("hopefulness", "Positive Expectancy", "hopefulness", "total_expectancies"),
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
    if "highthreat" in name or "_ace" in name:
        return "ACE"
    if "moderatethreat" in name or "moderate" in name:
        return "Moderate"
    if "lowthreat" in name or "default" in name or "noace" in name:
        return "No ACE"
    return ""


def parse_run_rows(csv_path: str, needed_cols: List[str]) -> List[Tuple[int, Dict[str, float]]]:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    run_header_idx = None
    for i, row in enumerate(rows):
        if not row:
            continue
        if row[0].strip().lower() == "run":
            header_set = {c.strip() for c in row}
            if set(needed_cols + ["Run"]).issubset(header_set):
                run_header_idx = i
                break

    if run_header_idx is None:
        return []

    header = [h.strip() for h in rows[run_header_idx]]
    idx = {k: header.index(k) for k in needed_cols + ["Run"]}

    parsed: List[Tuple[int, Dict[str, float]]] = []
    for row in rows[run_header_idx + 1 :]:
        if not row:
            break
        first = row[0].strip().lower()
        if first in {"statistic", "total", "mean", "std", "min", "max"}:
            break
        try:
            run_id = int(float(row[idx["Run"]]))
            values = {c: float(row[idx[c]]) for c in needed_cols}
        except (ValueError, IndexError):
            continue
        parsed.append((run_id, values))
    return parsed


def holm_adjust(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    m = len(p_values)
    out = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = (m - rank) * p
        running_max = max(running_max, val)
        out[orig_idx] = min(1.0, running_max)
    return out


def run_pipeline(files: List[str], threshold_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    needed_cols = TOTAL_COMPLEX_BEHAVIOR_COLUMNS

    run_rows = []
    for path in files:
        condition = infer_condition(path)
        if condition == "":
            continue
        parsed = parse_run_rows(path, needed_cols)
        for run_id, vals in parsed:
            total_actions = float(sum(vals[c] for c in TOTAL_COMPLEX_BEHAVIOR_COLUMNS))
            if total_actions <= 0:
                continue
            base = {
                "source_file": os.path.basename(path),
                "source_path": path,
                "condition": condition,
                "run": run_id,
                "total_actions": total_actions,
            }
            for b in TOTAL_COMPLEX_BEHAVIOR_COLUMNS:
                base[b] = float(vals[b])
            run_rows.append(base)

    if not run_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    runs_df = pd.DataFrame(run_rows)

    runs_df["total_complex_behaviors"] = runs_df[TOTAL_COMPLEX_BEHAVIOR_COLUMNS].sum(axis=1)
    runs_df["total_expectancies"] = runs_df[EXPECTANCY_DENOM_COLUMNS].sum(axis=1)

    # Percent per behavior per run with behavior-specific denominators.
    for behavior_key, _label, numerator_col, denom_col in ADAPTIVE_EMERGENCE_SPECS:
        denom = runs_df[denom_col].replace(0, np.nan)
        runs_df[f"{behavior_key}_pct"] = (runs_df[numerator_col] / denom) * 100.0
        runs_df[f"{behavior_key}_pct"] = runs_df[f"{behavior_key}_pct"].fillna(0)

    # Global pooled thresholds.
    th_rows = []
    thresholds = {}
    for b, label, _numerator_col, _denom_col in ADAPTIVE_EMERGENCE_SPECS:
        t = float(np.percentile(runs_df[f"{b}_pct"].values, threshold_pct))
        thresholds[b] = t
        th_rows.append({
            "behavior_key": b,
            "behavior": label,
            "threshold_percentile": threshold_pct,
            "threshold_percent": t,
        })

    # Presence labels.
    for b, _label, _numerator_col, _denom_col in ADAPTIVE_EMERGENCE_SPECS:
        t = thresholds[b]
        runs_df[f"{b}_present"] = (runs_df[f"{b}_pct"] >= t).astype(int)

    # Condition-level probabilities with 95% Wald CI.
    prob_rows = []
    for condition in CONDITIONS:
        sub = runs_df[runs_df["condition"] == condition]
        n = len(sub)
        if n == 0:
            continue
        for b, label, _numerator_col, _denom_col in ADAPTIVE_EMERGENCE_SPECS:
            p = float(sub[f"{b}_present"].mean())
            se = float(np.sqrt(max(p * (1.0 - p), 0.0) / n))
            ci = 1.96 * se
            prob_rows.append({
                "condition": condition,
                "behavior_key": b,
                "behavior": label,
                "n_runs": n,
                "n_present": int(sub[f"{b}_present"].sum()),
                "probability_percent": p * 100.0,
                "ci95_low_percent": max(0.0, (p - ci) * 100.0),
                "ci95_high_percent": min(100.0, (p + ci) * 100.0),
                "ci95_halfwidth_percent": ci * 100.0,
            })

    prob_df = pd.DataFrame(prob_rows)

    # Chi-square per behavior for condition x presence.
    stat_rows = []
    pvals = []
    keys = []
    for b, label, _numerator_col, _denom_col in ADAPTIVE_EMERGENCE_SPECS:
        contingency = []
        for condition in CONDITIONS:
            sub = runs_df[runs_df["condition"] == condition]
            present = int(sub[f"{b}_present"].sum())
            absent = int(len(sub) - present)
            contingency.append([present, absent])

        arr = np.array(contingency, dtype=float)
        if np.any(arr.sum(axis=1) == 0):
            chi2 = np.nan
            p = np.nan
            dof = np.nan
            cramers_v = np.nan
        else:
            try:
                chi2, p, dof, _ = chi2_contingency(arr)
                n_total = arr.sum()
                k = min(arr.shape[0] - 1, arr.shape[1] - 1)
                cramers_v = np.sqrt(chi2 / (n_total * k)) if n_total > 0 and k > 0 else np.nan
            except ValueError:
                chi2 = np.nan
                p = np.nan
                dof = np.nan
                cramers_v = np.nan

        stat_rows.append({
            "behavior_key": b,
            "behavior": label,
            "chi2": chi2,
            "p_value_raw": p,
            "dof": dof,
            "cramers_v": cramers_v,
            "significant_0_05": bool(p < 0.05) if not np.isnan(p) else False,
        })
        if not np.isnan(p):
            pvals.append(float(p))
            keys.append(b)

    if pvals:
        padj = holm_adjust(pvals)
        pmap = {k: v for k, v in zip(keys, padj)}
    else:
        pmap = {}

    for row in stat_rows:
        pa = pmap.get(row["behavior_key"], np.nan)
        row["p_value_holm"] = pa
        row["significant_holm_0_05"] = bool(pa < 0.05) if not np.isnan(pa) else False

    stats_df = pd.DataFrame(stat_rows)

    # Long raw table for transparency.
    raw_long = []
    for _, r in runs_df.iterrows():
        for b, label, numerator_col, denom_col in ADAPTIVE_EMERGENCE_SPECS:
            raw_long.append({
                "source_file": r["source_file"],
                "source_path": r["source_path"],
                "condition": r["condition"],
                "run": int(r["run"]),
                "behavior_key": b,
                "behavior": label,
                "behavior_count": float(r[numerator_col]),
                "total_actions": float(r[denom_col]),
                "behavior_percent": float(r[f"{b}_pct"]),
                "threshold_percent": float(thresholds[b]),
                "present": int(r[f"{b}_present"]),
            })
    raw_df = pd.DataFrame(raw_long)

    thresholds_df = pd.DataFrame(th_rows)
    return raw_df, thresholds_df, prob_df, stats_df


def plot_probability(prob_df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    behaviors = [label for _, label, _, _ in ADAPTIVE_EMERGENCE_SPECS]
    conditions = CONDITIONS
    x = np.arange(len(conditions), dtype=float)
    width = 0.14
    offsets = np.linspace(-2 * width, 2 * width, len(behaviors))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#FF9DA6"]

    for j, behavior in enumerate(behaviors):
        sub = prob_df[prob_df["behavior"] == behavior].set_index("condition")
        ys = [float(sub.loc[c, "probability_percent"]) if c in sub.index else 0.0 for c in conditions]
        err = [float(sub.loc[c, "ci95_halfwidth_percent"]) if c in sub.index else 0.0 for c in conditions]
        ax.bar(
            x + offsets[j],
            ys,
            width=width,
            label=behavior,
            color=colors[j % len(colors)],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            yerr=err,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of runs where behavior is present", fontsize=11, fontweight="bold")
    ax.set_xlabel("Condition", fontsize=11, fontweight="bold")
    ax.set_title("Figure 5B: Likelihood of adaptive behavior emergence by adversity", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
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
    parser = argparse.ArgumentParser(description="Figure 5B: Probability of Emergence of Adaptive Behaviors")
    parser.add_argument("--NB", nargs="*", default=["multiple_runs_NB_*Threat*.csv"], help="NB CSV files or globs")
    parser.add_argument("--DB", nargs="*", default=["multiple_runs_DB_*Threat*.csv"], help="DB CSV files or globs")
    parser.add_argument("--threshold", type=float, default=75.0, help="Global percentile threshold for emergence")
    parser.add_argument("--output-dir", default=os.path.join("FinalProject", "figure5"), help="Output folder")
    args = parser.parse_args()

    files = flatten_globs(args.DB) + flatten_globs(args.NB)
    if not files:
        raise SystemExit("No input files found. Provide --NB and/or --DB patterns.")

    raw_df, thresholds_df, prob_df, stats_df = run_pipeline(files, args.threshold)
    if raw_df.empty:
        raise SystemExit("No valid run rows found in input CSVs.")

    os.makedirs(args.output_dir, exist_ok=True)

    png_path = os.path.join(args.output_dir, "figure5b_adaptive_emergence_probability.png")
    raw_path = os.path.join(args.output_dir, "figure5b_raw_run_presence.csv")
    thresholds_path = os.path.join(args.output_dir, "figure5b_thresholds.csv")
    prob_path = os.path.join(args.output_dir, "figure5b_probability_by_condition.csv")
    stats_path = os.path.join(args.output_dir, "figure5b_stats_chi2.csv")

    plot_probability(prob_df, png_path)
    raw_saved = write_csv_with_fallback(raw_df, raw_path)
    thresholds_saved = write_csv_with_fallback(thresholds_df, thresholds_path)
    prob_saved = write_csv_with_fallback(prob_df, prob_path)
    stats_saved = write_csv_with_fallback(stats_df, stats_path)

    print(f"Saved plot: {png_path}")
    print(f"Saved raw run-level output: {raw_saved}")
    print(f"Saved thresholds: {thresholds_saved}")
    print(f"Saved condition probability table: {prob_saved}")
    print(f"Saved stats table: {stats_saved}")


if __name__ == "__main__":
    main()
