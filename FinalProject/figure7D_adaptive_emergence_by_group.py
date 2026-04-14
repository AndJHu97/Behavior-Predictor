#!/usr/bin/env python3
"""
Figure 7D: Adaptive behavior emergence by resilient/vulnerable groups.

Inputs:
- DB High Threat multiple-runs CSV files (run-level table)
- Figure 7A threshold file for resilient/vulnerable split

Behaviors and percentages:
- Protective behavior % = protective_behavior / total_complex_behaviors
- Positive expectancy % = hopefulness /
  (hopefulness + cynical + learned_helplessness)

Analysis:
1) Primary multivariate test: MANOVA on
   (protective_behavior_pct, positive_expectancy_pct) by group.
2) Follow-up individual tests per metric with Holm-Bonferroni and FDR-BH.
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro, ttest_ind

try:
    from statsmodels.multivariate.manova import MANOVA
    HAS_MANOVA = True
except ImportError:
    HAS_MANOVA = False


COMPLEX_BEHAVIOR_COLUMNS = [
    "protective_behavior",
    "community_trusting_vulnerability",
    "healthy_friendliness",
    "bully_behavior",
    "aggressive_withdrawn_relationship",
    "learned_helplessness",
    "dangerous_trust",
]

EXPECTANCY_COLUMNS = ["hopefulness", "cynical", "learned_helplessness"]

ADAPTIVE_SPECS = [
    ("protective_behavior_pct", "Protective behavior", "protective_behavior", "total_complex_behaviors"),
    (
        "positive_expectancy_pct",
        "Positive expectancy",
        "hopefulness",
        "hopefulness + cynical + learned_helplessness",
    ),
]


COLUMN_ALIASES = {
    "protective_behavior": ["protective_behavior"],
    "community_trusting_vulnerability": ["community_trusting_vulnerability"],
    "healthy_friendliness": ["healthy_friendliness"],
    "bully_behavior": ["bully_behavior"],
    "aggressive_withdrawn_relationship": ["aggressive_withdrawn_relationship"],
    "learned_helplessness": ["learned_helplessness", "Learned_helplessness"],
    "dangerous_trust": ["dangerous_trust"],
    "hopefulness": ["hopefulness"],
    "cynical": ["cynical"],
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
    idx: Dict[str, int] = {}

    if "run" not in normalized:
        raise ValueError("Run column missing")
    idx["Run"] = normalized["run"]

    for col in canonical_cols:
        aliases = COLUMN_ALIASES[col]
        found = None
        for alias in aliases:
            k = alias.strip().lower()
            if k in normalized:
                found = normalized[k]
                break
        if found is None:
            raise ValueError(f"Required column missing: {col}")
        idx[col] = found

    return idx


def parse_run_rows(csv_path: str) -> List[Dict[str, float]]:
    needed_cols = COMPLEX_BEHAVIOR_COLUMNS + [c for c in EXPECTANCY_COLUMNS if c not in COMPLEX_BEHAVIOR_COLUMNS]

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header_idx = _find_run_header(rows)
    if header_idx < 0:
        return []

    header = [h.strip() for h in rows[header_idx]]
    idx = _resolve_columns(header, needed_cols)

    out: List[Dict[str, float]] = []
    for row in rows[header_idx + 1 :]:
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

        total_complex = float(sum(values[c] for c in COMPLEX_BEHAVIOR_COLUMNS))
        total_expectancy = float(values["hopefulness"] + values["cynical"] + values["learned_helplessness"])
        if total_complex <= 0 or total_expectancy <= 0:
            continue

        row_out: Dict[str, float] = {
            "run": run_id,
            "total_complex_behaviors": total_complex,
            "total_expectancy": total_expectancy,
        }
        for c, v in values.items():
            row_out[c] = v

        row_out["learned_helplessness_pct"] = (values["learned_helplessness"] / total_complex) * 100.0
        row_out["protective_behavior_pct"] = (values["protective_behavior"] / total_complex) * 100.0
        row_out["positive_expectancy_pct"] = (values["hopefulness"] / total_expectancy) * 100.0
        out.append(row_out)

    return out


def collect_rows(files: List[str]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    rows: List[Dict[str, float]] = []
    skipped: List[Tuple[str, str]] = []

    for path in files:
        try:
            parsed = parse_run_rows(path)
            if not parsed:
                skipped.append((path, "No valid run rows found"))
                continue
            for r in parsed:
                rows.append(
                    {
                        "source_file": os.path.basename(path),
                        "source_path": path,
                        **r,
                    }
                )
        except Exception as exc:
            skipped.append((path, str(exc)))

    if not rows:
        return pd.DataFrame(), skipped
    return pd.DataFrame(rows), skipped


def read_threshold(path: str) -> float:
    df = pd.read_csv(path)
    if "threshold_pct" not in df.columns or df.empty:
        raise ValueError("threshold_pct missing in threshold file")
    return float(df.loc[0, "threshold_pct"])


def holm_adjust(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    m = len(p_values)
    out = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        candidate = (m - rank) * p
        running_max = max(running_max, candidate)
        out[orig_idx] = min(1.0, running_max)
    return out


def fdr_bh_adjust(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx, p = indexed[i]
        rank = i + 1
        val = min(prev, (p * m) / rank)
        adjusted[idx] = min(1.0, val)
        prev = val
    return adjusted


def compare_groups(values_res: np.ndarray, values_vul: np.ndarray) -> Dict[str, float]:
    if len(values_res) < 2 or len(values_vul) < 2:
        return {
            "test_used": "insufficient_data",
            "test_stat": np.nan,
            "p_raw": np.nan,
            "normality_res_p": np.nan,
            "normality_vul_p": np.nan,
            "res_mean": float(np.mean(values_res)) if len(values_res) else np.nan,
            "vul_mean": float(np.mean(values_vul)) if len(values_vul) else np.nan,
            "res_median": float(np.median(values_res)) if len(values_res) else np.nan,
            "vul_median": float(np.median(values_vul)) if len(values_vul) else np.nan,
            "cohens_d": np.nan,
        }

    p_norm_res = float(shapiro(values_res).pvalue) if len(values_res) <= 5000 else np.nan
    p_norm_vul = float(shapiro(values_vul).pvalue) if len(values_vul) <= 5000 else np.nan
    use_t = bool((p_norm_res > 0.05) and (p_norm_vul > 0.05))

    if use_t:
        t = ttest_ind(values_res, values_vul, equal_var=False)
        test_used = "welch_t_test"
        stat = float(t.statistic)
        p_raw = float(t.pvalue)
    else:
        u = mannwhitneyu(values_res, values_vul, alternative="two-sided")
        test_used = "mann_whitney_u"
        stat = float(u.statistic)
        p_raw = float(u.pvalue)

    m1, m2 = float(np.mean(values_res)), float(np.mean(values_vul))
    s1, s2 = float(np.std(values_res, ddof=1)), float(np.std(values_vul, ddof=1))
    n1, n2 = len(values_res), len(values_vul)
    pooled = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(1, (n1 + n2 - 2)))
    d = (m2 - m1) / pooled if pooled > 0 else np.nan

    return {
        "test_used": test_used,
        "test_stat": stat,
        "p_raw": p_raw,
        "normality_res_p": p_norm_res,
        "normality_vul_p": p_norm_vul,
        "res_mean": m1,
        "vul_mean": m2,
        "res_median": float(np.median(values_res)),
        "vul_median": float(np.median(values_vul)),
        "cohens_d": float(d),
    }


def run_manova(df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    if not HAS_MANOVA:
        msg = "statsmodels MANOVA unavailable in environment."
        return msg, {"wilks_lambda": np.nan, "f_value": np.nan, "p_value": np.nan}

    work = df.copy()
    work["group_code"] = np.where(work["latent_group"] == "vulnerable", 1, 0)
    formula = "protective_behavior_pct + positive_expectancy_pct ~ group_code"

    try:
        mv = MANOVA.from_formula(formula, data=work)
        res = mv.mv_test()
        summary_text = str(res)

        wilks_lambda = np.nan
        f_value = np.nan
        p_value = np.nan
        try:
            stat_df = res.results["group_code"]["stat"]
            if "Wilks' lambda" in stat_df.index:
                wilks_row = stat_df.loc["Wilks' lambda"]
                wilks_lambda = float(wilks_row["Value"])
                f_value = float(wilks_row["F Value"])
                p_value = float(wilks_row["Pr > F"])
        except Exception:
            pass

        return summary_text, {
            "wilks_lambda": wilks_lambda,
            "f_value": f_value,
            "p_value": p_value,
        }
    except Exception as exc:
        msg = f"MANOVA failed: {exc}"
        return msg, {"wilks_lambda": np.nan, "f_value": np.nan, "p_value": np.nan}


def plot_violin(df: pd.DataFrame, out_png: str) -> None:
    behavior_plot_specs = [
        ("protective_behavior_pct", "Protective\nBehavior"),
        ("positive_expectancy_pct", "Positive\nExpectancy"),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(42)
    x_ticks = []
    x_labels = []
    position = 1
    offset = 0.18

    for col, label in behavior_plot_specs:
        res_vals = df[df["latent_group"] == "resilient"][col].to_numpy(dtype=float)
        vul_vals = df[df["latent_group"] == "vulnerable"][col].to_numpy(dtype=float)

        if len(res_vals) > 0:
            vp_res = ax.violinplot([res_vals], positions=[position - offset], widths=0.30, showmeans=True, showmedians=True)
            for b in vp_res["bodies"]:
                b.set_facecolor("#4C78A8")
                b.set_edgecolor("#2E2E2E")
                b.set_alpha(0.65)
            jitter = rng.uniform(-0.04, 0.04, len(res_vals))
            ax.scatter(np.full(len(res_vals), position - offset) + jitter, res_vals, s=10, color="#1A1A1A", alpha=0.35)

        if len(vul_vals) > 0:
            vp_vul = ax.violinplot([vul_vals], positions=[position + offset], widths=0.30, showmeans=True, showmedians=True)
            for b in vp_vul["bodies"]:
                b.set_facecolor("#E45756")
                b.set_edgecolor("#2E2E2E")
                b.set_alpha(0.65)
            jitter = rng.uniform(-0.04, 0.04, len(vul_vals))
            ax.scatter(np.full(len(vul_vals), position + offset) + jitter, vul_vals, s=10, color="#1A1A1A", alpha=0.35)

        x_ticks.append(position)
        x_labels.append(label)
        position += 1

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Percentage of complex behavior or expectancy")
    ax.set_title("Figure 7D: Adaptive behavior emergence by resilient-vulnerable groups")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ax.scatter([], [], color="#4C78A8", label="Resilient")
    ax.scatter([], [], color="#E45756", label="Vulnerable")
    ax.legend(frameon=False)

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
        alt = f"{root}_updated{ext}"
        df.to_csv(alt, index=False)
        return alt


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 7D adaptive emergence by subgroup")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["multiple_runs_Largerun_DB_HighThreat*.csv", "multiple_runs_largerun_DB_highThreat*.csv"],
        help="Input run CSVs or glob patterns",
    )
    parser.add_argument(
        "--threshold-file",
        default=os.path.join("FinalProject", "figure7", "figure7A_threshold_analysis.csv"),
        help="Threshold CSV from Figure 7A",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("FinalProject", "figure7"),
        help="Output directory",
    )
    args = parser.parse_args()

    files = flatten_globs(args.inputs)
    if not files:
        raise SystemExit("No input files found. Provide DB HighThreat CSVs via --inputs.")

    if not os.path.isfile(args.threshold_file):
        raise SystemExit(f"Threshold file not found: {args.threshold_file}")

    threshold = read_threshold(args.threshold_file)
    raw_df, skipped = collect_rows(files)
    if raw_df.empty:
        raise SystemExit("No valid run-level rows parsed from inputs.")

    raw_df = raw_df.copy()
    raw_df["threshold_from_figure7A_pct"] = threshold
    raw_df["latent_group"] = np.where(raw_df["learned_helplessness_pct"] < threshold, "resilient", "vulnerable")

    followup_rows = []
    for behavior_key, label, _numerator, denominator_label in ADAPTIVE_SPECS:
        res_vals = raw_df[raw_df["latent_group"] == "resilient"][behavior_key].to_numpy(dtype=float)
        vul_vals = raw_df[raw_df["latent_group"] == "vulnerable"][behavior_key].to_numpy(dtype=float)
        stats = compare_groups(res_vals, vul_vals)
        followup_rows.append(
            {
                "behavior_key": behavior_key,
                "behavior": label,
                "denominator": denominator_label,
                "n_resilient": len(res_vals),
                "n_vulnerable": len(vul_vals),
                **stats,
            }
        )

    followup_df = pd.DataFrame(followup_rows)
    pvals = [float(p) if pd.notna(p) else np.nan for p in followup_df["p_raw"].tolist()]
    valid_idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    valid_p = [pvals[i] for i in valid_idx]

    holm_adj = [np.nan] * len(pvals)
    fdr_adj = [np.nan] * len(pvals)
    if valid_p:
        h = holm_adjust(valid_p)
        f = fdr_bh_adjust(valid_p)
        for j, i in enumerate(valid_idx):
            holm_adj[i] = h[j]
            fdr_adj[i] = f[j]

    followup_df["p_holm_bonferroni"] = holm_adj
    followup_df["p_fdr_bh"] = fdr_adj
    followup_df["significant_holm_0_05"] = followup_df["p_holm_bonferroni"] < 0.05
    followup_df["significant_fdr_0_05"] = followup_df["p_fdr_bh"] < 0.05

    manova_text, manova_metrics = run_manova(raw_df)

    desc_rows = []
    for behavior_key, label, _numerator, denominator_label in ADAPTIVE_SPECS:
        for grp in ["resilient", "vulnerable"]:
            vals = raw_df[raw_df["latent_group"] == grp][behavior_key].to_numpy(dtype=float)
            desc_rows.append(
                {
                    "behavior_key": behavior_key,
                    "behavior": label,
                    "denominator": denominator_label,
                    "group": grp,
                    "n": len(vals),
                    "mean_pct": float(np.mean(vals)) if len(vals) else np.nan,
                    "std_pct": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                    "median_pct": float(np.median(vals)) if len(vals) else np.nan,
                    "q1_pct": float(np.percentile(vals, 25)) if len(vals) else np.nan,
                    "q3_pct": float(np.percentile(vals, 75)) if len(vals) else np.nan,
                }
            )
    desc_df = pd.DataFrame(desc_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    fig_path = os.path.join(args.output_dir, "figure7D_adaptive_emergence_violin.png")
    raw_path = os.path.join(args.output_dir, "figure7D_raw_run_percentages.csv")
    desc_path = os.path.join(args.output_dir, "figure7D_descriptive_stats.csv")
    followup_path = os.path.join(args.output_dir, "figure7D_followup_tests.csv")
    manova_txt_path = os.path.join(args.output_dir, "figure7D_manova_results.txt")
    manova_metrics_path = os.path.join(args.output_dir, "figure7D_manova_metrics.csv")
    skipped_path = os.path.join(args.output_dir, "figure7D_skipped_files.csv")
    summary_path = os.path.join(args.output_dir, "figure7D_analysis_summary.txt")

    plot_violin(raw_df, fig_path)
    raw_saved = write_csv_with_fallback(raw_df, raw_path)
    desc_saved = write_csv_with_fallback(desc_df, desc_path)
    followup_saved = write_csv_with_fallback(followup_df, followup_path)
    manova_metrics_saved = write_csv_with_fallback(pd.DataFrame([manova_metrics]), manova_metrics_path)
    skipped_df = pd.DataFrame(skipped, columns=["source_path", "reason"]) if skipped else pd.DataFrame(columns=["source_path", "reason"])
    skipped_saved = write_csv_with_fallback(skipped_df, skipped_path)

    with open(manova_txt_path, "w", encoding="utf-8") as f:
        f.write("MANOVA on adaptive percentages\n")
        f.write("Dependent variables: protective_behavior_pct, positive_expectancy_pct\n")
        f.write("Predictor: latent_group (resilient/vulnerable)\n\n")
        f.write(manova_text)
        f.write("\n")

    summary_lines = [
        "Figure 7D adaptive emergence by group summary",
        "=" * 72,
        f"Input files found: {len(files)}",
        f"Valid runs analyzed: {len(raw_df)}",
        f"Threshold from Figure 7A: {threshold:.6f}%",
        f"Resilient runs: {int((raw_df['latent_group'] == 'resilient').sum())}",
        f"Vulnerable runs: {int((raw_df['latent_group'] == 'vulnerable').sum())}",
        "",
        "Primary test (MANOVA):",
        f"  Wilks lambda: {manova_metrics['wilks_lambda']}",
        f"  F value: {manova_metrics['f_value']}",
        f"  p value: {manova_metrics['p_value']}",
        "",
        "Follow-up tests saved with Holm-Bonferroni and FDR-BH corrections.",
        "",
        "Output files:",
        f"  Figure: {fig_path}",
        f"  Raw run table: {raw_saved}",
        f"  Descriptive stats: {desc_saved}",
        f"  Follow-up tests: {followup_saved}",
        f"  MANOVA text: {manova_txt_path}",
        f"  MANOVA metrics: {manova_metrics_saved}",
        f"  Skipped-file log: {skipped_saved}",
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved figure: {fig_path}")
    print(f"Saved raw output: {raw_saved}")
    print(f"Saved descriptive stats: {desc_saved}")
    print(f"Saved follow-up tests: {followup_saved}")
    print(f"Saved MANOVA text: {manova_txt_path}")
    print(f"Saved MANOVA metrics: {manova_metrics_saved}")
    print(f"Saved skipped-file log: {skipped_saved}")
    print(f"Saved summary report: {summary_path}")


if __name__ == "__main__":
    main()
