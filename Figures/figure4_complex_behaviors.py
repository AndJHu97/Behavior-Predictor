import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = ["No ACE", "Moderate adversity", "ACE"]

# Included for encoding consistency with simulation.py variable names.
ADAPTIVE_BEHAVIORS = {
    "Protective Aggression": "protective_behavior",
    "Adaptive Avoidance": "willingness_to_flee",
    "Prosocial Vulnerability": "community_trusting_vulnerability",
    "Affiliative Behavior": "healthy_friendliness",
    "Positive Expectancy Bias": "hopefulness",
}

MALADAPTIVE_BEHAVIORS = {
    "Hypervigilant Withdrawal": "fearful_withdrawn_relationship",
    "Dangerous Trust": "dangerous_trust",
    "Misdirected Aggression": "bully_behavior",
    "Relational Aggression": "aggressive_withdrawn_relationship",
    "Learned Helplessness": "learned_helplessness",
    "Negative Expectancy Bias": "cynical",
}

ACTION_COUNT_COLUMNS = [
    "fight_count",
    "flee_count",
    "befriend_count",
    "chase_count",
    "cry_count",
]

POSITIVE_EXPECTANCY_COLUMN = "hopefulness"
NEGATIVE_EXPECTANCY_COLUMN = "cynical"
LEARNED_HELPLESSNESS_COLUMN = "learned_helplessness"
EXPECTANCY_DENOM_COLUMNS = [POSITIVE_EXPECTANCY_COLUMN, NEGATIVE_EXPECTANCY_COLUMN, LEARNED_HELPLESSNESS_COLUMN]
ALL_COMPLEX_BEHAVIOR_COLUMNS = list(dict.fromkeys(list(ADAPTIVE_BEHAVIORS.values()) + list(MALADAPTIVE_BEHAVIORS.values())))
COMPLEX_ACTION_COLUMNS = [c for c in ALL_COMPLEX_BEHAVIOR_COLUMNS if c not in [POSITIVE_EXPECTANCY_COLUMN, NEGATIVE_EXPECTANCY_COLUMN]]

CONDITION_COLORS = {
    "No ACE": "#4C78A8",
    "Moderate adversity": "#F58518",
    "ACE": "#E45756",
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


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_run_rows(csv_path, required_cols):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    run_header_idx = None
    for i, row in enumerate(rows):
        if not row:
            continue
        if row[0].strip().lower() == "run":
            header_set = set([c.strip() for c in row])
            needed = set(required_cols + ALL_COMPLEX_BEHAVIOR_COLUMNS + ["Run"])
            if needed.issubset(header_set):
                run_header_idx = i
                break

    if run_header_idx is None:
        return None

    header = [h.strip() for h in rows[run_header_idx]]
    idx_map = {name: header.index(name) for name in required_cols + ALL_COMPLEX_BEHAVIOR_COLUMNS + ["Run"]}

    run_data = []
    for row in rows[run_header_idx + 1 :]:
        if not row:
            break
        first = row[0].strip().lower()
        if first in {"statistic", "total", "mean", "std", "min", "max"}:
            break
        if len(row) <= idx_map["Run"]:
            continue
        try:
            run_id = int(float(row[idx_map["Run"]]))
            values = {col: float(row[idx_map[col]]) for col in required_cols + ALL_COMPLEX_BEHAVIOR_COLUMNS}
        except (ValueError, IndexError):
            continue
        run_data.append((run_id, values))

    return run_data


def collect_figure4_rows(files):
    required_cols = list(MALADAPTIVE_BEHAVIORS.values())
    raw_rows = []
    skipped = []

    for csv_path in files:
        condition = infer_condition_from_filename(csv_path)
        if condition is None:
            skipped.append((csv_path, "Could not infer condition from filename"))
            continue

        run_data = parse_run_rows(csv_path, required_cols)
        if not run_data:
            skipped.append((csv_path, "Missing or invalid Run table for required behavior columns"))
            continue

        data_group = "Random" if "random" in os.path.basename(csv_path).lower() else "Observed"

        for run_id, values in run_data:
            total_complex_behaviors = sum(values[c] for c in COMPLEX_ACTION_COLUMNS)
            total_expectancies = sum(values[c] for c in EXPECTANCY_DENOM_COLUMNS)
            for behavior_label, behavior_col in MALADAPTIVE_BEHAVIORS.items():
                if behavior_col == NEGATIVE_EXPECTANCY_COLUMN:
                    # Negative expectancy includes cynical + learned helplessness.
                    behavior_count = values[NEGATIVE_EXPECTANCY_COLUMN] + values[LEARNED_HELPLESSNESS_COLUMN]
                else:
                    behavior_count = values[behavior_col]

                if behavior_col == NEGATIVE_EXPECTANCY_COLUMN:
                    denominator = total_expectancies
                    denominator_type = "total_expectancies"
                else:
                    denominator = total_complex_behaviors
                    denominator_type = "total_complex_behaviors"

                pct_of_actions = safe_percent(behavior_count, denominator)
                raw_rows.append(
                    {
                        "source_file": os.path.basename(csv_path),
                        "source_path": csv_path,
                        "data_group": data_group,
                        "condition": condition,
                        "run": run_id,
                        "behavior": behavior_label,
                        "behavior_column": behavior_col,
                        "denominator_type": denominator_type,
                        "behavior_count": behavior_count,
                        "total_actions": denominator,
                        "percent_of_total_actions": pct_of_actions,
                    }
                )

    return raw_rows, skipped


def median_iqr(values):
    if len(values) == 0:
        return "", "", ""
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return median, q1, q3


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


def run_figure4_stats(raw_rows):
    df = pd.DataFrame(raw_rows)
    out_rows = []

    for behavior in MALADAPTIVE_BEHAVIORS.keys():
        beh_df = df[df["behavior"] == behavior]

        groups = {}
        for condition in CONDITIONS:
            values = beh_df[beh_df["condition"] == condition]["percent_of_total_actions"].dropna().values
            groups[condition] = values

        median_no_ace, q1_no_ace, q3_no_ace = median_iqr(groups["No ACE"])
        median_mod, q1_mod, q3_mod = median_iqr(groups["Moderate adversity"])
        median_ace, q1_ace, q3_ace = median_iqr(groups["ACE"])

        # Kruskal-Wallis across all three conditions.
        if all(len(groups[c]) > 0 for c in CONDITIONS):
            h_stat, p_kruskal = stats.kruskal(groups["No ACE"], groups["Moderate adversity"], groups["ACE"])
            h_stat = float(h_stat)
            p_kruskal = float(p_kruskal)
        else:
            h_stat, p_kruskal = "", ""

        # Post-hoc pairwise (rank-based), focused on adversity contrasts requested.
        pairwise = []
        for comp, c1, c2 in [
            ("No_ACE_vs_ACE", "No ACE", "ACE"),
            ("Moderate_adversity_vs_ACE", "Moderate adversity", "ACE"),
        ]:
            if len(groups[c1]) > 0 and len(groups[c2]) > 0:
                u_stat, p_raw = stats.mannwhitneyu(groups[c1], groups[c2], alternative="two-sided")
                pairwise.append((comp, float(u_stat), float(p_raw)))
            else:
                pairwise.append((comp, "", ""))

        p_vals = [x[2] for x in pairwise if x[2] != ""]
        p_adj = holm_adjust(p_vals)
        p_iter = iter(p_adj)

        pairwise_out = {}
        for comp, u_stat, p_raw in pairwise:
            if p_raw == "":
                p_holm = ""
                sig = ""
            else:
                p_holm = next(p_iter)
                sig = bool(p_holm < 0.05)
            pairwise_out[f"{comp}_u_stat"] = u_stat
            pairwise_out[f"{comp}_p_raw"] = p_raw
            pairwise_out[f"{comp}_p_holm"] = p_holm
            pairwise_out[f"{comp}_significant_alpha_0_05"] = sig

        out_rows.append(
            {
                "behavior": behavior,
                "kruskal_h_stat": h_stat,
                "kruskal_p_value": p_kruskal,
                "kruskal_significant_alpha_0_05": bool(p_kruskal < 0.05) if p_kruskal != "" else "",
                "median_no_ace": median_no_ace,
                "q1_no_ace": q1_no_ace,
                "q3_no_ace": q3_no_ace,
                "median_moderate_adversity": median_mod,
                "q1_moderate_adversity": q1_mod,
                "q3_moderate_adversity": q3_mod,
                "median_ace": median_ace,
                "q1_ace": q1_ace,
                "q3_ace": q3_ace,
                "n_no_ace": len(groups["No ACE"]),
                "n_moderate_adversity": len(groups["Moderate adversity"]),
                "n_ace": len(groups["ACE"]),
                **pairwise_out,
            }
        )

    return out_rows


def plot_figure4(raw_rows, output_path):
    df = pd.DataFrame(raw_rows)
    behaviors = list(MALADAPTIVE_BEHAVIORS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    rng = np.random.default_rng(42)

    for idx, behavior in enumerate(behaviors):
        ax = axes[idx]
        x = np.arange(len(CONDITIONS), dtype=float)
        width = 0.34

        for i, condition in enumerate(CONDITIONS):
            observed_vals = (
                df[
                    (df["behavior"] == behavior)
                    & (df["condition"] == condition)
                    & (df["data_group"] == "Observed")
                ]["percent_of_total_actions"]
                .dropna()
                .values
            )
            random_vals = (
                df[
                    (df["behavior"] == behavior)
                    & (df["condition"] == condition)
                    & (df["data_group"] == "Random")
                ]["percent_of_total_actions"]
                .dropna()
                .values
            )

            if observed_vals.size > 0:
                obs_med = float(np.median(observed_vals))
                obs_q1 = float(np.percentile(observed_vals, 25))
                obs_q3 = float(np.percentile(observed_vals, 75))
                obs_x = x[i] - (width / 2)
                ax.bar(obs_x, obs_med, width=width, color=CONDITION_COLORS[condition], alpha=0.75, edgecolor="black", linewidth=1.2)
                ax.errorbar(obs_x, obs_med, yerr=[[obs_med - obs_q1], [obs_q3 - obs_med]], fmt="none", color="black", capsize=4, capthick=1.5, linewidth=1.5)
                obs_jitter = rng.normal(0, width * 0.12, size=observed_vals.size)
                ax.scatter(obs_x + obs_jitter, observed_vals, alpha=0.35, s=18, color=CONDITION_COLORS[condition], edgecolors="black", linewidth=0.4)

            if random_vals.size > 0:
                rnd_med = float(np.median(random_vals))
                rnd_q1 = float(np.percentile(random_vals, 25))
                rnd_q3 = float(np.percentile(random_vals, 75))
                rnd_x = x[i] + (width / 2)
                ax.bar(rnd_x, rnd_med, width=width, color="#cfcfcf", alpha=0.9, edgecolor="black", linewidth=1.2)
                ax.errorbar(rnd_x, rnd_med, yerr=[[rnd_med - rnd_q1], [rnd_q3 - rnd_med]], fmt="none", color="black", capsize=4, capthick=1.5, linewidth=1.5)
                rnd_jitter = rng.normal(0, width * 0.12, size=random_vals.size)
                ax.scatter(rnd_x + rnd_jitter, random_vals, alpha=0.35, s=18, color="#9a9a9a", edgecolors="black", linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, fontsize=11)
        if behavior in {"Negative Expectancy Bias", "Positive Expectancy Bias"}:
            ax.set_ylabel("% of Total Expectancies", fontsize=11, fontweight="bold")
        else:
            ax.set_ylabel("% of Total Complex Behaviors", fontsize=11, fontweight="bold")
        ax.set_title(behavior, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0)

        if idx == 0:
            from matplotlib.patches import Patch
            legend_items = [
                Patch(facecolor="#4C78A8", edgecolor="black", label="Observed (NB+DB)"),
                Patch(facecolor="#cfcfcf", edgecolor="black", label="Random baseline"),
            ]
            ax.legend(handles=legend_items, fontsize=10, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(stats_rows):
    print("\nFigure 4 summary (median % of total actions; IQR in brackets)")
    print("Behavior                              No ACE           Moderate         ACE")
    print("-" * 90)
    for row in stats_rows:
        no_ace = f"{row['median_no_ace']:.3f} [{row['q1_no_ace']:.3f}, {row['q3_no_ace']:.3f}]"
        mod = (
            f"{row['median_moderate_adversity']:.3f} "
            f"[{row['q1_moderate_adversity']:.3f}, {row['q3_moderate_adversity']:.3f}]"
        )
        ace = f"{row['median_ace']:.3f} [{row['q1_ace']:.3f}, {row['q3_ace']:.3f}]"
        print(f"{row['behavior'][:35]:<35} {no_ace:<18} {mod:<18} {ace:<18}")


def main():
    parser = argparse.ArgumentParser(
        description="Figure 4: Emergence of complex maladaptive behaviors across adversity conditions"
    )
    parser.add_argument("--NB", nargs="*", default=[], help="NB CSV files or globs")
    parser.add_argument("--DB", nargs="*", default=[], help="DB CSV files or globs")
    parser.add_argument("--RandomNB", nargs="*", default=[], help="Random NB CSV files or globs")
    parser.add_argument("--RandomDB", nargs="*", default=[], help="Random DB CSV files or globs")
    parser.add_argument(
        "--output",
        default=os.path.join("FinalProject", "figure4", "figure4_maladaptive_emergence.png"),
    )
    parser.add_argument(
        "--raw-output",
        default=os.path.join("FinalProject", "figure4", "figure4_raw_behavior_percentages.csv"),
    )
    parser.add_argument(
        "--stats-output",
        default=os.path.join("FinalProject", "figure4", "figure4_stats_kruskal.csv"),
    )

    args = parser.parse_args()

    nb_files = flatten_globs(args.NB)
    db_files = flatten_globs(args.DB)
    random_nb_files = flatten_globs(args.RandomNB)
    random_db_files = flatten_globs(args.RandomDB)
    files = db_files + nb_files + random_nb_files + random_db_files

    if not files:
        parser.error("No input files found. Provide files via --NB and/or --DB.")

    raw_rows, skipped = collect_figure4_rows(files)
    if not raw_rows:
        parser.error("No valid run rows were extracted from input files.")

    observed_rows = [r for r in raw_rows if r["data_group"] == "Observed"]
    stats_rows = run_figure4_stats(observed_rows)

    write_csv(
        args.raw_output,
        [
            "source_file",
            "source_path",
            "data_group",
            "condition",
            "run",
            "behavior",
            "behavior_column",
            "denominator_type",
            "behavior_count",
            "total_actions",
            "percent_of_total_actions",
        ],
        raw_rows,
    )

    write_csv(
        args.stats_output,
        [
            "behavior",
            "kruskal_h_stat",
            "kruskal_p_value",
            "kruskal_significant_alpha_0_05",
            "median_no_ace",
            "q1_no_ace",
            "q3_no_ace",
            "median_moderate_adversity",
            "q1_moderate_adversity",
            "q3_moderate_adversity",
            "median_ace",
            "q1_ace",
            "q3_ace",
            "n_no_ace",
            "n_moderate_adversity",
            "n_ace",
            "No_ACE_vs_ACE_u_stat",
            "No_ACE_vs_ACE_p_raw",
            "No_ACE_vs_ACE_p_holm",
            "No_ACE_vs_ACE_significant_alpha_0_05",
            "Moderate_adversity_vs_ACE_u_stat",
            "Moderate_adversity_vs_ACE_p_raw",
            "Moderate_adversity_vs_ACE_p_holm",
            "Moderate_adversity_vs_ACE_significant_alpha_0_05",
        ],
        stats_rows,
    )

    plot_figure4(raw_rows, args.output)

    if skipped:
        print("\nSkipped files:")
        for path, reason in skipped:
            print(f"- {path}: {reason}")

    print_summary(stats_rows)
    print(f"\nSaved Figure 4 to: {args.output}")
    print(f"Saved Figure 4 raw data to: {args.raw_output}")
    print(f"Saved Figure 4 stats to: {args.stats_output}")


if __name__ == "__main__":
    main()
