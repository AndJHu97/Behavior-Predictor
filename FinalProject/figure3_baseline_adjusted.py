import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

ACTIONS = ["fight_count", "flee_count", "cry_count", "chase_count"]
CONDITIONS = ["No ACE", "Moderate adversity", "ACE"]
ROLES = ["DB", "NB"]

ACTION_LABELS = {
    "fight_count": "Fight",
    "flee_count": "Flee",
    "cry_count": "Cry",
    "chase_count": "Chase",
}

CONDITION_COLORS = {
    "No ACE": "#4C78A8",
    "Moderate adversity": "#F58518",
    "ACE": "#E45756",
}

ROLE_COLORS = {
    "DB": "#1f77b4",
    "NB": "#ff7f0e",
}

RUN_COLUMNS = ["fight_count", "flee_count", "befriend_count", "chase_count", "cry_count"]

# Color palette: condition × role
FIG3_COLORS = {
    ("DB", "No ACE"): "#1f77b4",
    ("DB", "Moderate adversity"): "#1f77b4",
    ("DB", "ACE"): "#1f77b4",
    ("NB", "No ACE"): "#ff7f0e",
    ("NB", "Moderate adversity"): "#ff7f0e",
    ("NB", "ACE"): "#ff7f0e",
}

# Opacity variations by condition
CONDITION_OPACITY = {
    "No ACE": 0.6,
    "Moderate adversity": 0.8,
    "ACE": 1.0,
}


def flatten_globs(patterns):
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    return list(dict.fromkeys(files))


def infer_condition_from_filename(path):
    name = os.path.basename(path).lower()
    if "highthreat" in name or "_ace" in name:
        return "ACE"
    if "moderatethreat" in name or "moderate" in name:
        return "Moderate adversity"
    if "lowthreat" in name or "noace" in name or "default_model" in name:
        return "No ACE"
    return None


def safe_percent(numerator, denominator):
    return (numerator / denominator) * 100.0 if denominator else 0.0


def mean(values):
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return sum(values) / len(values) if len(values) > 0 else 0.0


def ci95_half_width(values):
    if len(values) <= 1:
        return 0.0
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    return float(1.96 * sem)


def holm_adjust(p_values):
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    m = len(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (idx, p_val) in enumerate(indexed):
        candidate = (m - rank) * p_val
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_run_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    run_header_idx = None
    for i, row in enumerate(rows):
        if not row:
            continue
        if row[0].strip().lower() == "run" and all(c in row for c in RUN_COLUMNS):
            run_header_idx = i
            break

    if run_header_idx is None:
        return None

    header = [h.strip() for h in rows[run_header_idx]]
    idx_map = {name: header.index(name) for name in RUN_COLUMNS + ["Run"]}

    run_data = []
    for row in rows[run_header_idx + 1 :]:
        if not row:
            break
        first = row[0].strip().lower() if row else ""
        if first in {"statistic", "total", "mean", "std", "min", "max"}:
            break
        if len(row) <= idx_map["Run"]:
            continue
        try:
            run_id = int(float(row[idx_map["Run"]]))
            values = {col: float(row[idx_map[col]]) for col in RUN_COLUMNS}
        except (ValueError, IndexError):
            continue
        run_data.append((run_id, values))

    return run_data


def collect_baseline_and_role_rows(db_files, nb_files, random_files):
    """Collect Random baseline and role data, aligned by condition."""
    # Parse Random first
    random_by_condition = {c: [] for c in CONDITIONS}
    random_skipped = []
    for csv_path in random_files:
        run_data = parse_run_rows(csv_path)
        if not run_data:
            random_skipped.append((csv_path, "Missing or invalid Run table"))
            continue
        for _, values in run_data:
            action_total = sum(values[a] for a in ACTIONS)
            for action in ACTIONS:
                pct = safe_percent(values[action], action_total)
                random_by_condition["No ACE"].append(pct)  # Random is condition-independent

    # Consolidate random percentages into single distribution
    random_all_percentages = {a: [] for a in ACTIONS}
    for csv_path in random_files:
        run_data = parse_run_rows(csv_path)
        if run_data:
            for _, values in run_data:
                action_total = sum(values[a] for a in ACTIONS)
                for action in ACTIONS:
                    pct = safe_percent(values[action], action_total)
                    random_all_percentages[action].append(pct)

    random_baseline_pcts = {a: mean(random_all_percentages[a]) for a in ACTIONS}

    # Parse DB and NB by role and condition
    role_condition_data = {}
    role_skipped = []

    for role_name, role_files in [("DB", db_files), ("NB", nb_files)]:
        for condition in CONDITIONS:
            role_condition_data[(role_name, condition)] = {a: [] for a in ACTIONS}

        for csv_path in role_files:
            condition = infer_condition_from_filename(csv_path)
            if condition is None:
                role_skipped.append((csv_path, "Could not infer condition from filename"))
                continue

            run_data = parse_run_rows(csv_path)
            if not run_data:
                role_skipped.append((csv_path, "Missing or invalid Run table"))
                continue

            for _, values in run_data:
                action_total = sum(values[a] for a in ACTIONS)
                for action in ACTIONS:
                    pct = safe_percent(values[action], action_total)
                    role_condition_data[(role_name, condition)][action].append(pct)

    return random_baseline_pcts, role_condition_data, random_skipped + role_skipped


def collect_baseline_adjusted_rows(db_files, nb_files, random_files):
    """Build raw rows with baseline-adjusted (delta) percentages."""
    random_baseline_pcts, role_condition_data, skipped = collect_baseline_and_role_rows(
        db_files, nb_files, random_files
    )

    raw_rows = []

    for role_name, role_files in [("DB", db_files), ("NB", nb_files)]:
        for csv_path in role_files:
            condition = infer_condition_from_filename(csv_path)
            if condition is None:
                continue

            run_data = parse_run_rows(csv_path)
            if not run_data:
                continue

            for run_id, values in run_data:
                action_total = sum(values[a] for a in ACTIONS)
                row = {
                    "source_file": os.path.basename(csv_path),
                    "source_path": csv_path,
                    "role": role_name,
                    "condition": condition,
                    "run": run_id,
                }
                for action in ACTIONS:
                    pct = safe_percent(values[action], action_total)
                    delta = pct - random_baseline_pcts[action]  # Baseline-adjusted
                    row[action] = delta
                raw_rows.append(row)

    return raw_rows, random_baseline_pcts, skipped


def aggregate_baseline_adjusted(raw_rows):
    """Aggregate delta percentages by role and condition."""
    df = pd.DataFrame(raw_rows)

    mean_data = {}
    ci95_data = {}

    for role in ROLES:
        for condition in CONDITIONS:
            subset = df[(df["role"] == role) & (df["condition"] == condition)]
            mean_data[(role, condition)] = {}
            ci95_data[(role, condition)] = {}

            for action in ACTIONS:
                if not subset.empty:
                    vals = subset[action].dropna().values
                    mean_data[(role, condition)][action] = mean(vals) if vals.size > 0 else 0.0
                    ci95_data[(role, condition)][action] = ci95_half_width(vals) if vals.size > 0 else 0.0
                else:
                    mean_data[(role, condition)][action] = 0.0
                    ci95_data[(role, condition)][action] = 0.0

    return mean_data, ci95_data


def run_figure3_clustered_stats(raw_rows):
    """GEE model for baseline-adjusted deviations: delta ~ role + condition + role*condition."""
    df = pd.DataFrame(raw_rows)
    out_rows = []

    for action in ACTIONS:
        model_df = df[["role", "condition", "source_file", action]].copy().dropna()

        formula = (
            f"{action} ~ C(role, Treatment(reference='DB')) "
            "+ C(condition, Treatment(reference='No ACE')) "
            "+ C(role, Treatment(reference='DB')):C(condition, Treatment(reference='No ACE'))"
        )

        gee = GEE.from_formula(
            formula,
            groups="source_file",
            cov_struct=Exchangeable(),
            family=Gaussian(),
            data=model_df,
        )
        fit = gee.fit()

        param_names = list(fit.params.index)

        # Extract main effect Wald tests
        role_terms = [n for n in param_names if n.startswith("C(role") and ":" not in n]
        cond_terms = [n for n in param_names if n.startswith("C(condition") and ":" not in n]
        int_terms = [n for n in param_names if ":" in n]

        def _safe_float(value):
            try:
                arr = np.asarray(value).squeeze()
                if np.size(arr) == 1:
                    return float(arr)
                return ""
            except Exception:
                return ""

        def _wald_subset(fit, param_names, selected_names):
            if not selected_names:
                return "", ""
            matrix = []
            for term in selected_names:
                row = [0.0] * len(param_names)
                row[param_names.index(term)] = 1.0
                matrix.append(row)
            res = fit.wald_test(matrix, scalar=True)
            return _safe_float(res.statistic), _safe_float(res.pvalue)

        def _contrast_from_terms(fit, param_names, weights):
            vec = [0.0] * len(param_names)
            for name, coef in weights.items():
                if name in param_names:
                    vec[param_names.index(name)] = coef
            tt = fit.t_test(vec)
            return _safe_float(tt.effect), _safe_float(tt.sd), _safe_float(tt.tvalue), _safe_float(tt.pvalue)

        role_stat, role_p = _wald_subset(fit, param_names, role_terms)
        cond_stat, cond_p = _wald_subset(fit, param_names, cond_terms)
        int_stat, int_p = _wald_subset(fit, param_names, int_terms)

        out_rows.append(
            {
                "action": action,
                "analysis": "effect_tests",
                "role_main_wald": role_stat,
                "role_main_p": role_p,
                "condition_main_wald": cond_stat,
                "condition_main_p": cond_p,
                "interaction_wald": int_stat,
                "interaction_p": int_p,
                "n_runs": len(model_df),
                "n_clusters": model_df["source_file"].nunique(),
            }
        )

    return out_rows


def run_figure3_delta_vs_zero_tests(raw_rows):
    """One-sample tests: test if Δ differs from 0 for each action, role, condition."""
    df = pd.DataFrame(raw_rows)
    out_rows = []

    for action in ACTIONS:
        for role in ROLES:
            for condition in CONDITIONS:
                subset = df[(df["role"] == role) & (df["condition"] == condition)]
                if subset.empty:
                    continue

                values = subset[action].dropna().values
                if len(values) < 2:
                    continue

                # One-sample t-test: H0: mean = 0
                t_stat, p_val = stats.ttest_1samp(values, 0)
                mean_delta = float(np.mean(values))
                sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
                ci_lower = mean_delta - 1.96 * sem
                ci_upper = mean_delta + 1.96 * sem
                significant = bool(p_val < 0.05)

                out_rows.append(
                    {
                        "action": action,
                        "role": role,
                        "condition": condition,
                        "mean_delta": mean_delta,
                        "sem": sem,
                        "ci_lower_95": ci_lower,
                        "ci_upper_95": ci_upper,
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "significant_alpha_0_05": significant,
                        "n_runs": len(values),
                    }
                )

    return out_rows


def run_figure3_role_comparisons(raw_rows):
    """Two-sample tests: compare Δ between DB and NB for each action and condition."""
    df = pd.DataFrame(raw_rows)
    out_rows = []

    for action in ACTIONS:
        for condition in CONDITIONS:
            db_subset = df[(df["role"] == "DB") & (df["condition"] == condition)]
            nb_subset = df[(df["role"] == "NB") & (df["condition"] == condition)]

            if db_subset.empty or nb_subset.empty:
                continue

            db_values = db_subset[action].dropna().values
            nb_values = nb_subset[action].dropna().values

            if len(db_values) < 2 or len(nb_values) < 2:
                continue

            # Two-sample t-test: H0: DB_mean = NB_mean
            t_stat, p_val = stats.ttest_ind(db_values, nb_values)
            db_mean = float(np.mean(db_values))
            nb_mean = float(np.mean(nb_values))
            diff = db_mean - nb_mean

            # Pooled SEM for difference
            db_sem = np.std(db_values, ddof=1) / np.sqrt(len(db_values))
            nb_sem = np.std(nb_values, ddof=1) / np.sqrt(len(nb_values))
            diff_sem = float(np.sqrt(db_sem ** 2 + nb_sem ** 2))
            ci_lower = diff - 1.96 * diff_sem
            ci_upper = diff + 1.96 * diff_sem
            significant = bool(p_val < 0.05)

            out_rows.append(
                {
                    "action": action,
                    "condition": condition,
                    "db_mean_delta": db_mean,
                    "nb_mean_delta": nb_mean,
                    "difference": diff,
                    "diff_sem": diff_sem,
                    "ci_lower_95": ci_lower,
                    "ci_upper_95": ci_upper,
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "significant_alpha_0_05": significant,
                    "n_db": len(db_values),
                    "n_nb": len(nb_values),
                }
            )

    return out_rows


def plot_baseline_adjusted(mean_data, ci95_data, output_path, random_baseline_pcts=None):
    """Plot baseline-adjusted deviations grouped by role and condition."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = list(range(len(ACTIONS)))
    width = 0.12
    
    # 6 bars per action: DB No ACE, DB Moderate, DB ACE, NB No ACE, NB Moderate, NB ACE
    role_condition_pairs = [
        ("DB", "No ACE"),
        ("DB", "Moderate adversity"),
        ("DB", "ACE"),
        ("NB", "No ACE"),
        ("NB", "Moderate adversity"),
        ("NB", "ACE"),
    ]

    for i, (role, condition) in enumerate(role_condition_pairs):
        bar_x = [v + (i - 2.5) * width for v in x]
        heights = [mean_data[(role, condition)][action] for action in ACTIONS]
        errs = [ci95_data[(role, condition)][action] for action in ACTIONS]

        # Color based on role and opacity based on condition
        base_color = ROLE_COLORS[role]
        alpha = CONDITION_OPACITY[condition]

        ax.bar(
            bar_x,
            heights,
            yerr=errs,
            capsize=3,
            width=width,
            color=base_color,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.5,
            label=f"{role} – {condition}",
        )

    # Add horizontal line at y=0 to show baseline
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random baseline")

    ax.set_title(
        "Figure 3: Deviation from Random Baseline Reveals Learned Behavior",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ACTION_LABELS[action] for action in ACTIONS])
    ax.set_ylabel("Δ % (Observed - Random)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary_figure3(mean_data, random_baseline_pcts):
    print("\nFigure 3: Baseline-Adjusted Behavior (Δ % = Observed - Random)")
    print("Random Baseline: Fight={:.2f}%, Flee={:.2f}%, Cry={:.2f}%, Chase={:.2f}%".format(
        random_baseline_pcts["fight_count"],
        random_baseline_pcts["flee_count"],
        random_baseline_pcts["cry_count"],
        random_baseline_pcts["chase_count"],
    ))
    print("\nBaseline-Adjusted Deviations (%):")
    print("Role  Condition           Fight    Flee     Cry      Chase")
    print("-" * 70)
    for role in ROLES:
        for condition in CONDITIONS:
            print(
                f"{role:<4} {condition:<18} "
                f"{mean_data[(role, condition)]['fight_count']:>7.2f} "
                f"{mean_data[(role, condition)]['flee_count']:>7.2f} "
                f"{mean_data[(role, condition)]['cry_count']:>7.2f} "
                f"{mean_data[(role, condition)]['chase_count']:>7.2f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Figure 3: Baseline-adjusted behavior (delta from random) with run-level clustered GEE stats"
    )
    parser.add_argument("--DB", nargs="*", default=[], help="DB CSV files or globs")
    parser.add_argument("--NB", nargs="*", default=[], help="NB CSV files or globs")
    parser.add_argument("--Random", nargs="*", default=[], help="Random CSV files or globs")
    parser.add_argument(
        "--output",
        default=os.path.join("FinalProject", "figure3", "figure3_baseline_adjusted.png"),
    )
    parser.add_argument(
        "--raw-output",
        default=os.path.join("FinalProject", "figure3", "figure3_raw_deltas.csv"),
    )
    parser.add_argument(
        "--stats-output",
        default=os.path.join("FinalProject", "figure3", "figure3_stats_clustered.csv"),
    )
    parser.add_argument(
        "--delta-vs-zero-output",
        default=os.path.join("FinalProject", "figure3", "figure3_delta_vs_zero.csv"),
        help="One-sample tests: is Δ significantly different from 0?",
    )
    parser.add_argument(
        "--role-comparison-output",
        default=os.path.join("FinalProject", "figure3", "figure3_db_vs_nb_deltas.csv"),
        help="Two-sample tests: does DB Δ differ from NB Δ?",
    )

    args = parser.parse_args()

    db_files = flatten_globs(args.DB)
    nb_files = flatten_globs(args.NB)
    random_files = flatten_globs(args.Random)

    if not db_files or not nb_files:
        parser.error("Must provide DB and NB files. Provide --DB and --NB arguments.")
    if not random_files:
        parser.error("Must provide Random baseline. Provide --Random argument.")

    raw_rows, random_baseline_pcts, skipped = collect_baseline_adjusted_rows(db_files, nb_files, random_files)
    
    if not raw_rows:
        parser.error("No valid rows collected from input files.")

    mean_data, ci95_data = aggregate_baseline_adjusted(raw_rows)
    stats_rows = run_figure3_clustered_stats(raw_rows)
    delta_vs_zero_rows = run_figure3_delta_vs_zero_tests(raw_rows)
    role_comparison_rows = run_figure3_role_comparisons(raw_rows)

    write_csv(
        args.raw_output,
        ["source_file", "source_path", "role", "condition", "run"] + ACTIONS,
        raw_rows,
    )
    write_csv(
        args.stats_output,
        [
            "action",
            "analysis",
            "role_main_wald",
            "role_main_p",
            "condition_main_wald",
            "condition_main_p",
            "interaction_wald",
            "interaction_p",
            "n_runs",
            "n_clusters",
        ],
        stats_rows,
    )

    write_csv(
        args.delta_vs_zero_output,
        [
            "action",
            "role",
            "condition",
            "mean_delta",
            "sem",
            "ci_lower_95",
            "ci_upper_95",
            "t_statistic",
            "p_value",
            "significant_alpha_0_05",
            "n_runs",
        ],
        delta_vs_zero_rows,
    )

    write_csv(
        args.role_comparison_output,
        [
            "action",
            "condition",
            "db_mean_delta",
            "nb_mean_delta",
            "difference",
            "diff_sem",
            "ci_lower_95",
            "ci_upper_95",
            "t_statistic",
            "p_value",
            "significant_alpha_0_05",
            "n_db",
            "n_nb",
        ],
        role_comparison_rows,
    )

    if skipped:
        print("\nSkipped files:")
        for path, reason in skipped:
            print(f"- {path}: {reason}")

    print_summary_figure3(mean_data, random_baseline_pcts)

    plot_baseline_adjusted(mean_data, ci95_data, args.output, random_baseline_pcts)

    print(f"\nSaved Figure 3 to: {args.output}")
    print(f"Saved Figure 3 raw data to: {args.raw_output}")
    print(f"Saved Figure 3 GEE effect tests to: {args.stats_output}")
    print(f"Saved Figure 3 delta vs zero tests to: {args.delta_vs_zero_output}")
    print(f"Saved Figure 3 role comparisons to: {args.role_comparison_output}")


if __name__ == "__main__":
    main()
