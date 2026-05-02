#!/usr/bin/env python3
"""
Figure 7B: Protective factors versus learned helplessness (DB High Threat).

Uses the same DB High Threat run-level CSV format as Figure 7A and applies
the resilient/vulnerable split using the threshold estimated in Figure 7A.

Panel A:
- X: learned helplessness percentage among 7 complex behaviors
- Y: healthy friendliness percentage among 7 complex behaviors

Panel B:
- X: learned helplessness percentage among 7 complex behaviors
- Y: help-seeking vulnerability percentage (community_trusting_vulnerability)

The script computes:
- Linear regression and correlation on the full dataset
- Group difference between resilient and vulnerable
  (Welch t-test if normality assumptions are acceptable, otherwise Mann-Whitney)
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, mannwhitneyu, shapiro, spearmanr, ttest_ind


COMPLEX_BEHAVIOR_COLUMNS = [
    "protective_behavior",
    "community_trusting_vulnerability",
    "healthy_friendliness",
    "bully_behavior",
    "aggressive_withdrawn_relationship",
    "learned_helplessness",
    "dangerous_trust",
]

ENVIRONMENT_COLUMNS = ["threat_count", "ally_count", "prey_count"]


COLUMN_ALIASES = {
    "protective_behavior": ["protective_behavior"],
    "community_trusting_vulnerability": ["community_trusting_vulnerability"],
    "healthy_friendliness": ["healthy_friendliness"],
    "bully_behavior": ["bully_behavior"],
    "aggressive_withdrawn_relationship": ["aggressive_withdrawn_relationship"],
    "learned_helplessness": ["learned_helplessness", "Learned_helplessness"],
    "dangerous_trust": ["dangerous_trust"],
    "threat_count": ["threat_count"],
    "ally_count": ["ally_count"],
    "prey_count": ["prey_count"],
}


def flatten_globs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    return list(dict.fromkeys(files))


def _find_run_header(rows: List[List[str]]) -> int:
    for i, row in enumerate(rows):
        if row and row[0].strip().lower() == "run":
            return i
    return -1


def _resolve_columns(header: List[str], canonical_cols: List[str]) -> Dict[str, int]:
    normalized = {h.strip().lower(): i for i, h in enumerate(header)}
    out: Dict[str, int] = {}

    if "run" not in normalized:
        raise ValueError("Run column missing")
    out["Run"] = normalized["run"]

    for col in canonical_cols:
        aliases = COLUMN_ALIASES[col]
        matched = None
        for alias in aliases:
            key = alias.strip().lower()
            if key in normalized:
                matched = normalized[key]
                break
        if matched is None:
            raise ValueError(f"Required column missing: {col}")
        out[col] = matched

    return out


def parse_run_rows(csv_path: str) -> List[Dict[str, float]]:
    required_cols = COMPLEX_BEHAVIOR_COLUMNS + ENVIRONMENT_COLUMNS

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    run_header_idx = _find_run_header(rows)
    if run_header_idx < 0:
        return []

    header = [h.strip() for h in rows[run_header_idx]]
    idx = _resolve_columns(header, required_cols)

    parsed: List[Dict[str, float]] = []
    for row in rows[run_header_idx + 1 :]:
        if not row:
            break
        first = row[0].strip().lower()
        if first in {"statistic", "total", "mean", "std", "min", "max"}:
            break

        try:
            run_id = int(float(row[idx["Run"]]))
            values = {c: float(row[idx[c]]) for c in required_cols}
        except (ValueError, IndexError):
            continue

        total_complex = float(sum(values[c] for c in COMPLEX_BEHAVIOR_COLUMNS))
        total_env = float(sum(values[c] for c in ENVIRONMENT_COLUMNS))

        if total_complex <= 0 or total_env <= 0:
            continue

        lh_pct = (values["learned_helplessness"] / total_complex) * 100.0
        healthy_pct = (values["healthy_friendliness"] / total_complex) * 100.0
        help_seeking_vulnerability_pct = (values["community_trusting_vulnerability"] / total_complex) * 100.0
        ally_env_pct = (values["ally_count"] / total_env) * 100.0

        parsed.append(
            {
                "run": run_id,
                "total_complex_behaviors": total_complex,
                "total_environment_count": total_env,
                "learned_helplessness_pct": lh_pct,
                "healthy_friendliness_pct": healthy_pct,
                "help_seeking_vulnerability_pct": help_seeking_vulnerability_pct,
                "ally_environment_pct": ally_env_pct,
                **values,
            }
        )

    return parsed


def collect_rows(files: List[str]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    rows: List[Dict[str, float]] = []
    skipped: List[Tuple[str, str]] = []

    for path in files:
        try:
            parsed = parse_run_rows(path)
            if not parsed:
                skipped.append((path, "No valid run rows found"))
                continue
            for row in parsed:
                rows.append(
                    {
                        "source_file": os.path.basename(path),
                        "source_path": path,
                        **row,
                    }
                )
        except Exception as exc:
            skipped.append((path, str(exc)))

    if not rows:
        return pd.DataFrame(), skipped
    return pd.DataFrame(rows), skipped


def read_threshold(threshold_csv: str) -> float:
    df = pd.read_csv(threshold_csv)
    if "threshold_pct" not in df.columns or df.empty:
        raise ValueError("threshold_pct column missing or empty in threshold file")
    return float(df.loc[0, "threshold_pct"])


def regression_and_correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {
            "n": len(x),
            "slope": np.nan,
            "intercept": np.nan,
            "rvalue": np.nan,
            "r_squared": np.nan,
            "regression_pvalue": np.nan,
            "regression_stderr": np.nan,
            "spearman_rho": np.nan,
            "spearman_pvalue": np.nan,
        }

    reg = linregress(x, y)
    rho, p_s = spearmanr(x, y)
    return {
        "n": len(x),
        "slope": float(reg.slope),
        "intercept": float(reg.intercept),
        "rvalue": float(reg.rvalue),
        "r_squared": float(reg.rvalue ** 2),
        "regression_pvalue": float(reg.pvalue),
        "regression_stderr": float(reg.stderr),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(p_s),
    }


def compare_groups(resilient: np.ndarray, vulnerable: np.ndarray) -> Dict[str, object]:
    if len(resilient) < 2 or len(vulnerable) < 2:
        return {
            "test_used": "insufficient_data",
            "pvalue": np.nan,
            "statistic": np.nan,
            "normality_resilient_p": np.nan,
            "normality_vulnerable_p": np.nan,
            "resilient_mean": float(np.mean(resilient)) if len(resilient) else np.nan,
            "vulnerable_mean": float(np.mean(vulnerable)) if len(vulnerable) else np.nan,
            "resilient_median": float(np.median(resilient)) if len(resilient) else np.nan,
            "vulnerable_median": float(np.median(vulnerable)) if len(vulnerable) else np.nan,
            "cohens_d": np.nan,
            "cliffs_delta": np.nan,
        }

    normality_res = shapiro(resilient).pvalue if len(resilient) <= 5000 else np.nan
    normality_vul = shapiro(vulnerable).pvalue if len(vulnerable) <= 5000 else np.nan

    use_ttest = bool((normality_res > 0.05) and (normality_vul > 0.05))

    if use_ttest:
        t = ttest_ind(resilient, vulnerable, equal_var=False)
        statistic = float(t.statistic)
        pvalue = float(t.pvalue)
        test_used = "welch_t_test"
    else:
        u = mannwhitneyu(resilient, vulnerable, alternative="two-sided")
        statistic = float(u.statistic)
        pvalue = float(u.pvalue)
        test_used = "mann_whitney_u"

    # Cohen's d
    m1, m2 = float(np.mean(resilient)), float(np.mean(vulnerable))
    s1, s2 = float(np.std(resilient, ddof=1)), float(np.std(vulnerable, ddof=1))
    n1, n2 = len(resilient), len(vulnerable)
    pooled = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(n1 + n2 - 2, 1))
    cohens_d = (m2 - m1) / pooled if pooled > 0 else np.nan

    # Cliff's delta
    greater = 0
    lower = 0
    for rv in resilient:
        greater += int(np.sum(rv > vulnerable))
        lower += int(np.sum(rv < vulnerable))
    cliffs_delta = (greater - lower) / float(n1 * n2)

    return {
        "test_used": test_used,
        "pvalue": pvalue,
        "statistic": statistic,
        "normality_resilient_p": float(normality_res),
        "normality_vulnerable_p": float(normality_vul),
        "resilient_mean": m1,
        "vulnerable_mean": m2,
        "resilient_median": float(np.median(resilient)),
        "vulnerable_median": float(np.median(vulnerable)),
        "cohens_d": float(cohens_d),
        "cliffs_delta": float(cliffs_delta),
    }


def draw_scatter_with_fit(
    ax,
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
) -> None:
    res = df[df["latent_group"] == "resilient"]
    vul = df[df["latent_group"] == "vulnerable"]

    ax.scatter(
        res["learned_helplessness_pct"],
        res[y_col],
        color="#4C78A8",
        alpha=0.75,
        s=35,
        label=f"Resilient (n={len(res)})",
    )
    ax.scatter(
        vul["learned_helplessness_pct"],
        vul[y_col],
        color="#E45756",
        alpha=0.75,
        s=35,
        label=f"Vulnerable (n={len(vul)})",
    )

    x = df["learned_helplessness_pct"].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        reg = linregress(x, y)
        r2 = reg.rvalue ** 2
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ys = reg.intercept + reg.slope * xs
        ax.plot(
            xs,
            ys,
            color="#222222",
            linewidth=2.0,
            linestyle="--",
            label=f"Linear fit (all runs, R^2={r2:.3f})",
        )

    ax.set_xlabel("Learned helplessness (% of total complex behaviors)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=9)


def write_csv_with_fallback(df: pd.DataFrame, path: str) -> str:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        alt = f"{root}_updated{ext}"
        df.to_csv(alt, index=False)
        return alt


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 7B protective factors versus learned helplessness")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["multiple_runs_Largerun_DB_HighThreat*.csv", "multiple_runs_largerun_DB_highThreat*.csv"],
        help="Input CSV files or glob patterns",
    )
    parser.add_argument(
        "--threshold-file",
        default=os.path.join("FinalProject", "figure7", "figure7A_threshold_analysis.csv"),
        help="Threshold CSV generated by Figure 7A",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("FinalProject", "figure7"),
        help="Output directory",
    )
    args = parser.parse_args()

    files = flatten_globs(args.inputs)
    if not files:
        raise SystemExit("No input files found. Provide DB High Threat run CSVs via --inputs.")

    if not os.path.isfile(args.threshold_file):
        raise SystemExit(f"Threshold file not found: {args.threshold_file}")

    threshold = read_threshold(args.threshold_file)

    raw_df, skipped = collect_rows(files)
    if raw_df.empty:
        raise SystemExit("No valid run-level rows parsed from inputs.")

    raw_df = raw_df.copy()
    raw_df["threshold_from_figure7A_pct"] = threshold
    raw_df["latent_group"] = np.where(
        raw_df["learned_helplessness_pct"] < threshold,
        "resilient",
        "vulnerable",
    )

    # Analyses for Panel A: Healthy friendliness vs LH.
    x = raw_df["learned_helplessness_pct"].to_numpy(dtype=float)
    y_healthy = raw_df["healthy_friendliness_pct"].to_numpy(dtype=float)
    reg_healthy = regression_and_correlation(x, y_healthy)

    healthy_res = raw_df[raw_df["latent_group"] == "resilient"]["healthy_friendliness_pct"].to_numpy(dtype=float)
    healthy_vul = raw_df[raw_df["latent_group"] == "vulnerable"]["healthy_friendliness_pct"].to_numpy(dtype=float)
    diff_healthy = compare_groups(healthy_res, healthy_vul)

    # Analyses for Panel B: Help-seeking vulnerability vs LH.
    y_help_seek = raw_df["help_seeking_vulnerability_pct"].to_numpy(dtype=float)
    reg_help_seek = regression_and_correlation(x, y_help_seek)

    help_seek_res = raw_df[raw_df["latent_group"] == "resilient"]["help_seeking_vulnerability_pct"].to_numpy(dtype=float)
    help_seek_vul = raw_df[raw_df["latent_group"] == "vulnerable"]["help_seeking_vulnerability_pct"].to_numpy(dtype=float)
    diff_help_seek = compare_groups(help_seek_res, help_seek_vul)

    # Figure with two scatter panels.
    os.makedirs(args.output_dir, exist_ok=True)
    fig_path = os.path.join(args.output_dir, "figure7B_protective_factors_vs_lh.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw_scatter_with_fit(
        axes[0],
        raw_df,
        "healthy_friendliness_pct",
        "Healthy friendliness (% of total complex behaviors)",
        "Panel A: Healthy friendliness vs learned helplessness",
    )
    draw_scatter_with_fit(
        axes[1],
        raw_df,
        "help_seeking_vulnerability_pct",
        "Help-seeking vulnerability (% of total complex behaviors)",
        "Panel B: Help-seeking vulnerability vs learned helplessness",
    )
    fig.suptitle("Figure 7B: Protective factors by resilient-vulnerable split", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Output tables.
    raw_path = os.path.join(args.output_dir, "figure7B_raw_run_protective_factors.csv")
    stat_corr_path = os.path.join(args.output_dir, "figure7B_correlation_regression.csv")
    stat_group_path = os.path.join(args.output_dir, "figure7B_group_difference_tests.csv")
    skipped_path = os.path.join(args.output_dir, "figure7B_skipped_files.csv")
    summary_path = os.path.join(args.output_dir, "figure7B_analysis_summary.txt")

    corr_df = pd.DataFrame(
        [
            {
                "panel": "A",
                "outcome": "healthy_friendliness_pct",
                **reg_healthy,
            },
            {
                "panel": "B",
                "outcome": "help_seeking_vulnerability_pct",
                **reg_help_seek,
            },
        ]
    )

    group_df = pd.DataFrame(
        [
            {
                "panel": "A",
                "outcome": "healthy_friendliness_pct",
                "n_resilient": len(healthy_res),
                "n_vulnerable": len(healthy_vul),
                **diff_healthy,
            },
            {
                "panel": "B",
                "outcome": "help_seeking_vulnerability_pct",
                "n_resilient": len(help_seek_res),
                "n_vulnerable": len(help_seek_vul),
                **diff_help_seek,
            },
        ]
    )

    raw_saved = write_csv_with_fallback(raw_df, raw_path)
    corr_saved = write_csv_with_fallback(corr_df, stat_corr_path)
    group_saved = write_csv_with_fallback(group_df, stat_group_path)
    skipped_df = pd.DataFrame(skipped, columns=["source_path", "reason"]) if skipped else pd.DataFrame(columns=["source_path", "reason"])
    skipped_saved = write_csv_with_fallback(skipped_df, skipped_path)

    summary_lines = [
        "Figure 7B protective factor analysis summary",
        "=" * 72,
        f"Input files found: {len(files)}",
        f"Valid runs analyzed: {len(raw_df)}",
        f"Threshold from Figure 7A: {threshold:.6f}%",
        f"Resilient runs: {int((raw_df['latent_group'] == 'resilient').sum())}",
        f"Vulnerable runs: {int((raw_df['latent_group'] == 'vulnerable').sum())}",
        "",
        "Panel A (Healthy friendliness %):",
        f"  Regression slope: {reg_healthy['slope']:.6f}",
        f"  Pearson r: {reg_healthy['rvalue']:.6f}, R^2: {reg_healthy['r_squared']:.6f}, p: {reg_healthy['regression_pvalue']:.6g}",
        f"  Spearman rho: {reg_healthy['spearman_rho']:.6f}, p: {reg_healthy['spearman_pvalue']:.6g}",
        f"  Group test: {diff_healthy['test_used']}, stat: {diff_healthy['statistic']:.6f}, p: {diff_healthy['pvalue']:.6g}",
        "",
        "Panel B (Help-seeking vulnerability %):",
        f"  Regression slope: {reg_help_seek['slope']:.6f}",
        f"  Pearson r: {reg_help_seek['rvalue']:.6f}, R^2: {reg_help_seek['r_squared']:.6f}, p: {reg_help_seek['regression_pvalue']:.6g}",
        f"  Spearman rho: {reg_help_seek['spearman_rho']:.6f}, p: {reg_help_seek['spearman_pvalue']:.6g}",
        f"  Group test: {diff_help_seek['test_used']}, stat: {diff_help_seek['statistic']:.6f}, p: {diff_help_seek['pvalue']:.6g}",
        "",
        "Output files:",
        f"  Figure: {fig_path}",
        f"  Raw table: {raw_saved}",
        f"  Correlation and regression stats: {corr_saved}",
        f"  Group-difference stats: {group_saved}",
        f"  Skipped-file log: {skipped_saved}",
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved figure: {fig_path}")
    print(f"Saved raw output: {raw_saved}")
    print(f"Saved correlation/regression analysis: {corr_saved}")
    print(f"Saved group-difference analysis: {group_saved}")
    print(f"Saved skipped-file log: {skipped_saved}")
    print(f"Saved summary report: {summary_path}")


if __name__ == "__main__":
    main()
