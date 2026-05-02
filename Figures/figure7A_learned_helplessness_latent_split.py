#!/usr/bin/env python3
"""
Figure 7A: Learned helplessness latent subgroup split (DB High Threat)

Pipeline:
1. Read run-level rows from multiple_runs_DB_HighThreat*.csv style files.
2. Compute learned helplessness percentage among 7 complex behaviors per run:
   learned_helplessness / total_complex_behaviors * 100.
3. Fit Gaussian Mixture Models (1-component and 2-component).
4. Compare model fit via AIC, BIC, and log-likelihood.
5. Use posterior probability from 2-component GMM to find threshold where
   posterior(high-learned-helplessness component) is approximately 0.5.
6. Split runs into resilient (below threshold) and vulnerable (at/above threshold).
7. Produce a 2-panel figure:
   Panel A: Histogram + KDE of learned helplessness percentage.
   Panel B: Violin plot for All, Resilient, Vulnerable.

Outputs are written to FinalProject/figure7 by default.
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


BEHAVIOR_COLUMNS = [
    "protective_behavior",
    "community_trusting_vulnerability",
    "healthy_friendliness",
    "bully_behavior",
    "aggressive_withdrawn_relationship",
    "learned_helplessness",
    "dangerous_trust",
]


COLUMN_ALIASES = {
    "protective_behavior": ["protective_behavior"],
    "community_trusting_vulnerability": ["community_trusting_vulnerability"],
    "healthy_friendliness": ["healthy_friendliness"],
    "bully_behavior": ["bully_behavior"],
    "aggressive_withdrawn_relationship": ["aggressive_withdrawn_relationship"],
    "learned_helplessness": ["learned_helplessness", "Learned_helplessness"],
    "dangerous_trust": ["dangerous_trust"],
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


def _find_header_index(rows: List[List[str]]) -> int:
    for i, row in enumerate(rows):
        if not row:
            continue
        if row[0].strip().lower() == "run":
            return i
    return -1


def _resolve_columns(header: List[str]) -> Dict[str, int]:
    normalized = {h.strip().lower(): i for i, h in enumerate(header)}
    idx_map: Dict[str, int] = {}

    if "run" not in normalized:
        raise ValueError("Run column not found in run-level table header.")
    idx_map["Run"] = normalized["run"]

    for canonical in BEHAVIOR_COLUMNS:
        aliases = COLUMN_ALIASES[canonical]
        found = None
        for alias in aliases:
            key = alias.strip().lower()
            if key in normalized:
                found = normalized[key]
                break
        if found is None:
            raise ValueError(f"Required behavior column missing: {canonical}")
        idx_map[canonical] = found

    return idx_map


def parse_run_rows(csv_path: str) -> List[Dict[str, float]]:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    run_header_idx = _find_header_index(rows)
    if run_header_idx < 0:
        return []

    header = [h.strip() for h in rows[run_header_idx]]
    idx = _resolve_columns(header)

    parsed: List[Dict[str, float]] = []
    for row in rows[run_header_idx + 1 :]:
        if not row:
            break
        first = row[0].strip().lower()
        if first in {"statistic", "total", "mean", "std", "min", "max"}:
            break
        try:
            run_id = int(float(row[idx["Run"]]))
            values = {b: float(row[idx[b]]) for b in BEHAVIOR_COLUMNS}
        except (ValueError, IndexError):
            continue

        total_complex = float(sum(values[b] for b in BEHAVIOR_COLUMNS))
        if total_complex <= 0:
            continue

        learned_helplessness_pct = (values["learned_helplessness"] / total_complex) * 100.0

        parsed.append(
            {
                "run": run_id,
                "total_complex_behaviors": total_complex,
                "learned_helplessness_pct": learned_helplessness_pct,
                **values,
            }
        )

    return parsed


def collect_rows(files: List[str]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    out_rows: List[Dict[str, float]] = []
    skipped: List[Tuple[str, str]] = []

    for path in files:
        try:
            run_rows = parse_run_rows(path)
            if not run_rows:
                skipped.append((path, "No valid run-level rows found"))
                continue

            for row in run_rows:
                out_rows.append(
                    {
                        "source_file": os.path.basename(path),
                        "source_path": path,
                        **row,
                    }
                )
        except Exception as exc:  # Keep pipeline robust across mixed files.
            skipped.append((path, str(exc)))

    if not out_rows:
        return pd.DataFrame(), skipped

    df = pd.DataFrame(out_rows)
    return df, skipped


def fit_gmm_models(values: np.ndarray, random_state: int = 42) -> Dict[str, object]:
    x = values.reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, covariance_type="full", random_state=random_state, n_init=10)
    gmm2 = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state, n_init=10)

    gmm1.fit(x)
    gmm2.fit(x)

    ll1_total = float(gmm1.score(x) * len(x))
    ll2_total = float(gmm2.score(x) * len(x))

    aic1 = float(gmm1.aic(x))
    aic2 = float(gmm2.aic(x))
    bic1 = float(gmm1.bic(x))
    bic2 = float(gmm2.bic(x))

    means = gmm2.means_.reshape(-1)
    stds = np.sqrt(gmm2.covariances_.reshape(-1))
    weights = gmm2.weights_.reshape(-1)

    order = np.argsort(means)
    low_idx = int(order[0])
    high_idx = int(order[1])

    separation_s = float(abs(means[high_idx] - means[low_idx]) / (stds[high_idx] + stds[low_idx]))

    lo = max(0.0, float(np.min(values)) - 1.0)
    hi = min(100.0, float(np.max(values)) + 1.0)
    if hi <= lo:
        lo, hi = 0.0, 100.0

    grid = np.linspace(lo, hi, 5000)
    post = gmm2.predict_proba(grid.reshape(-1, 1))
    p_high = post[:, high_idx]
    idx = int(np.argmin(np.abs(p_high - 0.5)))
    threshold = float(grid[idx])
    posterior_at_threshold = float(p_high[idx])

    two_component_preferred = bool((aic2 < aic1) and (bic2 < bic1) and (ll2_total > ll1_total))

    return {
        "gmm1": gmm1,
        "gmm2": gmm2,
        "metrics": {
            "aic_1": aic1,
            "aic_2": aic2,
            "bic_1": bic1,
            "bic_2": bic2,
            "log_likelihood_1": ll1_total,
            "log_likelihood_2": ll2_total,
            "delta_aic_2_minus_1": aic2 - aic1,
            "delta_bic_2_minus_1": bic2 - bic1,
            "delta_log_likelihood_2_minus_1": ll2_total - ll1_total,
            "two_component_preferred": two_component_preferred,
        },
        "component": {
            "low_mean": float(means[low_idx]),
            "low_std": float(stds[low_idx]),
            "low_weight": float(weights[low_idx]),
            "high_mean": float(means[high_idx]),
            "high_std": float(stds[high_idx]),
            "high_weight": float(weights[high_idx]),
            "separation_s": separation_s,
        },
        "threshold": {
            "threshold_pct": threshold,
            "posterior_high_component": posterior_at_threshold,
            "grid_min": lo,
            "grid_max": hi,
        },
    }


def describe_separation(separation_s: float) -> str:
    if separation_s >= 1.0:
        return "clear_separation"
    if separation_s >= 0.5:
        return "moderate_overlap"
    return "substantial_overlap"


def make_figure(
    values: np.ndarray,
    threshold: float,
    resilient: np.ndarray,
    vulnerable: np.ndarray,
    out_png: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: histogram + KDE.
    ax = axes[0]
    bins = min(20, max(8, int(np.sqrt(len(values)))))
    ax.hist(values, bins=bins, density=True, alpha=0.5, color="#4C78A8", edgecolor="white")
    if len(values) >= 2 and np.std(values) > 0:
        kde = gaussian_kde(values)
        x_grid = np.linspace(max(0.0, np.min(values) - 1.0), min(100.0, np.max(values) + 1.0), 400)
        ax.plot(x_grid, kde(x_grid), color="#1F3552", linewidth=2.0)

    ax.axvline(threshold, color="#E45756", linestyle="--", linewidth=2.0)
    ax.set_xlabel("Learned helplessness (% of total complex behaviors)")
    ax.set_ylabel("Density")
    ax.set_title("Panel A: Distribution with KDE")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Panel B: violin split by inferred subgroup.
    ax = axes[1]
    groups = [values, resilient, vulnerable]
    labels = ["All", "Resilient", "Vulnerable"]
    colors = ["#9EC1D9", "#4C78A8", "#E45756"]

    plotted_positions = []
    plotted_data = []
    for i, g in enumerate(groups):
        if len(g) > 0:
            plotted_positions.append(i + 1)
            plotted_data.append(g)

    if plotted_data:
        vp = ax.violinplot(plotted_data, positions=plotted_positions, showmeans=True, showmedians=True)
        for j, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[plotted_positions[j] - 1])
            body.set_edgecolor("#2E2E2E")
            body.set_alpha(0.65)

    rng = np.random.default_rng(42)
    for i, g in enumerate(groups):
        if len(g) == 0:
            ax.text(i + 1, 1.0, "n=0", ha="center", va="bottom", fontsize=9, color="#666666")
            continue
        jitter = rng.uniform(-0.07, 0.07, size=len(g))
        ax.scatter(np.full(len(g), i + 1) + jitter, g, s=14, alpha=0.4, color="#1A1A1A")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Learned helplessness (% of total complex behaviors)")
    ax.set_title("Panel B: Latent subgroup split")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Figure 7A: Learned Helplessness Latent Subgroups (DB High Threat)", fontsize=13, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="Figure 7A: learned helplessness latent subgroup analysis")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=["multiple_runs_Largerun_DB_HighThreat*.csv", "multiple_runs_largerun_DB_highThreat*.csv"],
        help="Input CSV files or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("FinalProject", "figure7"),
        help="Output directory for figure and analysis files",
    )
    args = parser.parse_args()

    files = flatten_globs(args.inputs)
    if not files:
        raise SystemExit("No input files found. Provide --inputs with DB High Threat CSV files or glob patterns.")

    raw_df, skipped = collect_rows(files)
    if raw_df.empty:
        raise SystemExit("No valid run-level data parsed from input files.")

    values = raw_df["learned_helplessness_pct"].to_numpy(dtype=float)
    if len(values) < 3:
        raise SystemExit("Need at least 3 runs to fit and compare mixture models.")

    fit = fit_gmm_models(values)
    threshold = float(fit["threshold"]["threshold_pct"])

    raw_df = raw_df.copy()
    raw_df["latent_group"] = np.where(
        raw_df["learned_helplessness_pct"] < threshold,
        "resilient",
        "vulnerable",
    )

    resilient = raw_df[raw_df["latent_group"] == "resilient"]["learned_helplessness_pct"].to_numpy(dtype=float)
    vulnerable = raw_df[raw_df["latent_group"] == "vulnerable"]["learned_helplessness_pct"].to_numpy(dtype=float)

    separation_s = float(fit["component"]["separation_s"])
    separation_label = describe_separation(separation_s)

    model_comp_df = pd.DataFrame(
        [
            {
                "model": "1_component",
                "aic": fit["metrics"]["aic_1"],
                "bic": fit["metrics"]["bic_1"],
                "log_likelihood_total": fit["metrics"]["log_likelihood_1"],
            },
            {
                "model": "2_component",
                "aic": fit["metrics"]["aic_2"],
                "bic": fit["metrics"]["bic_2"],
                "log_likelihood_total": fit["metrics"]["log_likelihood_2"],
            },
        ]
    )

    model_decision_df = pd.DataFrame(
        [
            {
                "delta_aic_2_minus_1": fit["metrics"]["delta_aic_2_minus_1"],
                "delta_bic_2_minus_1": fit["metrics"]["delta_bic_2_minus_1"],
                "delta_log_likelihood_2_minus_1": fit["metrics"]["delta_log_likelihood_2_minus_1"],
                "two_component_preferred": fit["metrics"]["two_component_preferred"],
            }
        ]
    )

    component_df = pd.DataFrame(
        [
            {
                "component": "low_learned_helplessness",
                "mean_pct": fit["component"]["low_mean"],
                "std_pct": fit["component"]["low_std"],
                "weight": fit["component"]["low_weight"],
            },
            {
                "component": "high_learned_helplessness",
                "mean_pct": fit["component"]["high_mean"],
                "std_pct": fit["component"]["high_std"],
                "weight": fit["component"]["high_weight"],
            },
        ]
    )

    threshold_df = pd.DataFrame(
        [
            {
                "threshold_pct": fit["threshold"]["threshold_pct"],
                "posterior_high_component_at_threshold": fit["threshold"]["posterior_high_component"],
                "separation_s": separation_s,
                "separation_classification": separation_label,
            }
        ]
    )

    subgroup_df = pd.DataFrame(
        [
            {
                "group": "all",
                "n_runs": len(values),
                "mean_pct": float(np.mean(values)),
                "std_pct": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min_pct": float(np.min(values)),
                "max_pct": float(np.max(values)),
            },
            {
                "group": "resilient",
                "n_runs": len(resilient),
                "mean_pct": float(np.mean(resilient)) if len(resilient) > 0 else np.nan,
                "std_pct": float(np.std(resilient, ddof=1)) if len(resilient) > 1 else np.nan,
                "min_pct": float(np.min(resilient)) if len(resilient) > 0 else np.nan,
                "max_pct": float(np.max(resilient)) if len(resilient) > 0 else np.nan,
            },
            {
                "group": "vulnerable",
                "n_runs": len(vulnerable),
                "mean_pct": float(np.mean(vulnerable)) if len(vulnerable) > 0 else np.nan,
                "std_pct": float(np.std(vulnerable, ddof=1)) if len(vulnerable) > 1 else np.nan,
                "min_pct": float(np.min(vulnerable)) if len(vulnerable) > 0 else np.nan,
                "max_pct": float(np.max(vulnerable)) if len(vulnerable) > 0 else np.nan,
            },
        ]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    fig_path = os.path.join(args.output_dir, "figure7A_learned_helplessness_latent_split.png")
    raw_path = os.path.join(args.output_dir, "figure7A_raw_run_percentages.csv")
    model_comp_path = os.path.join(args.output_dir, "figure7A_model_comparison.csv")
    model_decision_path = os.path.join(args.output_dir, "figure7A_model_decision.csv")
    component_path = os.path.join(args.output_dir, "figure7A_gmm_components.csv")
    threshold_path = os.path.join(args.output_dir, "figure7A_threshold_analysis.csv")
    subgroup_path = os.path.join(args.output_dir, "figure7A_subgroup_summary.csv")
    skipped_path = os.path.join(args.output_dir, "figure7A_skipped_files.csv")
    analysis_path = os.path.join(args.output_dir, "figure7A_analysis_summary.txt")

    make_figure(values, threshold, resilient, vulnerable, fig_path)

    raw_saved = write_csv_with_fallback(raw_df, raw_path)
    model_comp_saved = write_csv_with_fallback(model_comp_df, model_comp_path)
    model_decision_saved = write_csv_with_fallback(model_decision_df, model_decision_path)
    component_saved = write_csv_with_fallback(component_df, component_path)
    threshold_saved = write_csv_with_fallback(threshold_df, threshold_path)
    subgroup_saved = write_csv_with_fallback(subgroup_df, subgroup_path)

    skipped_df = pd.DataFrame(skipped, columns=["source_path", "reason"]) if skipped else pd.DataFrame(columns=["source_path", "reason"])
    skipped_saved = write_csv_with_fallback(skipped_df, skipped_path)

    summary_lines = [
        "Figure 7A learned helplessness latent subgroup analysis",
        "=" * 72,
        f"Input files parsed: {len(files)}",
        f"Valid runs analyzed: {len(raw_df)}",
        "",
        "Model fit comparison:",
        f"  1-component: AIC={fit['metrics']['aic_1']:.3f}, BIC={fit['metrics']['bic_1']:.3f}, LogLik={fit['metrics']['log_likelihood_1']:.3f}",
        f"  2-component: AIC={fit['metrics']['aic_2']:.3f}, BIC={fit['metrics']['bic_2']:.3f}, LogLik={fit['metrics']['log_likelihood_2']:.3f}",
        f"  Preferred by AIC/BIC/LogLik rule: {fit['metrics']['two_component_preferred']}",
        "",
        "Two-component parameters:",
        f"  Low component mean={fit['component']['low_mean']:.3f}, std={fit['component']['low_std']:.3f}, weight={fit['component']['low_weight']:.3f}",
        f"  High component mean={fit['component']['high_mean']:.3f}, std={fit['component']['high_std']:.3f}, weight={fit['component']['high_weight']:.3f}",
        f"  Separation metric S=|mu1-mu2|/(sigma1+sigma2)={separation_s:.3f} ({separation_label})",
        "",
        "Posterior threshold:",
        f"  Threshold at posterior(high component) ~= 0.5: {threshold:.3f}%",
        f"  Posterior(high) at threshold grid point: {fit['threshold']['posterior_high_component']:.3f}",
        "",
        "Latent subgroup counts:",
        f"  Resilient (below threshold): {len(resilient)} runs",
        f"  Vulnerable (at/above threshold): {len(vulnerable)} runs",
        "",
        "Output files:",
        f"  Figure: {fig_path}",
        f"  Raw run table: {raw_saved}",
        f"  Model comparison: {model_comp_saved}",
        f"  Model decision: {model_decision_saved}",
        f"  Component table: {component_saved}",
        f"  Threshold table: {threshold_saved}",
        f"  Subgroup summary: {subgroup_saved}",
        f"  Skipped-file log: {skipped_saved}",
    ]

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved figure: {fig_path}")
    print(f"Saved raw run table: {raw_saved}")
    print(f"Saved model comparison table: {model_comp_saved}")
    print(f"Saved model decision table: {model_decision_saved}")
    print(f"Saved GMM component table: {component_saved}")
    print(f"Saved threshold analysis table: {threshold_saved}")
    print(f"Saved subgroup summary table: {subgroup_saved}")
    print(f"Saved skipped-file log: {skipped_saved}")
    print(f"Saved analysis summary: {analysis_path}")


if __name__ == "__main__":
    main()
