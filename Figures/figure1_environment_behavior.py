import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

ACTIONS = ["fight_count", "flee_count", "cry_count", "chase_count"]
FIG1B_ACTIONS = ["learned_helplessness", "apathy"]
CONDITIONS = ["Random", "No ACE", "Moderate adversity", "ACE"]
NON_RANDOM_CONDITIONS = ["No ACE", "Moderate adversity", "ACE"]

ACTION_LABELS = {
    "fight_count": "Fight",
    "flee_count": "Flee",
    "cry_count": "Cry",
    "chase_count": "Chase",
    "learned_helplessness": "Learned Helplessness",
    "apathy": "Apathy",
}

COLORS = {
    "Random": "#8c8c8c",
    "No ACE": "#4C78A8",
    "Moderate adversity": "#F58518",
    "ACE": "#E45756",
}

RUN_COLUMNS = [
    "fight_count",
    "flee_count",
    "befriend_count",
    "chase_count",
    "cry_count",
    "learned_helplessness",
    "apathy",
]


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
    return sum(values) / len(values) if values else 0.0


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


def extract_figure1_rows(nb_files, db_files, random_files):
    rows_1a = []
    rows_1b = []
    skipped = []

    def ingest(files, agent_type):
        for csv_path in files:
            if agent_type == "Random":
                condition = "Random"
            else:
                condition = infer_condition_from_filename(csv_path)
                if condition is None:
                    skipped.append((csv_path, "Could not infer condition from filename"))
                    continue

            run_data = parse_run_rows(csv_path)
            if not run_data:
                skipped.append((csv_path, "Missing or invalid Run table"))
                continue

            for run_id, values in run_data:
                action_total = sum(values[a] for a in ACTIONS)
                row_1a = {
                    "source_file": os.path.basename(csv_path),
                    "source_path": csv_path,
                    "agent_type": agent_type,
                    "condition": condition,
                    "run": run_id,
                }
                for a in ACTIONS:
                    row_1a[a] = safe_percent(values[a], action_total)
                rows_1a.append(row_1a)

                if agent_type != "Random":
                    lh_total = action_total + values["learned_helplessness"] + values["apathy"]
                    rows_1b.append(
                        {
                            "source_file": os.path.basename(csv_path),
                            "source_path": csv_path,
                            "agent_type": agent_type,
                            "condition": condition,
                            "run": run_id,
                            "learned_helplessness": safe_percent(
                                values["learned_helplessness"], lh_total
                            ),
                            "apathy": safe_percent(values["apathy"], lh_total),
                        }
                    )

    ingest(nb_files, "NB")
    ingest(db_files, "DB")
    ingest(random_files, "Random")
    return rows_1a, rows_1b, skipped


def aggregate_rows_to_means_and_ci95(rows, actions, conditions):
    mean_out = {}
    ci95_out = {}
    for condition in conditions:
        mean_out[condition] = {}
        ci95_out[condition] = {}
        for action in actions:
            vals = [r[action] for r in rows if r["condition"] == condition]
            mean_out[condition][action] = mean(vals)
            ci95_out[condition][action] = ci95_half_width(vals)
    return mean_out, ci95_out


def _safe_float(value):
    try:
        arr = np.asarray(value).squeeze()
        if np.size(arr) == 1:
            return float(arr)
        return ""
    except Exception:
        return ""


def run_figure1a_clustered_stats(rows_1a):
    df = pd.DataFrame(rows_1a)
    out_rows = []

    for action in ACTIONS:
        model_df = df[["condition", "source_file", action]].copy()
        model_df = model_df.dropna()

        formula = f"{action} ~ C(condition, Treatment(reference='Random'))"
        gee = GEE.from_formula(
            formula,
            groups="source_file",
            cov_struct=Exchangeable(),
            family=Gaussian(),
            data=model_df,
        )
        fit = gee.fit()

        params = fit.params
        cov = fit.cov_params()
        param_names = list(params.index)

        contrast_terms = [p for p in param_names if "C(condition" in p and "[T." in p]
        if contrast_terms:
            matrix = []
            for term in contrast_terms:
                row = [0.0] * len(param_names)
                row[param_names.index(term)] = 1.0
                matrix.append(row)
            global_res = fit.wald_test(matrix, scalar=True)
            global_test = "GEE Wald test"
            global_stat = _safe_float(global_res.statistic)
            global_p = _safe_float(global_res.pvalue)
        else:
            global_test = "GEE Wald test"
            global_stat = ""
            global_p = ""

        p_vals = []
        interim = []
        for cond in NON_RANDOM_CONDITIONS:
            term_name = f"C(condition, Treatment(reference='Random'))[T.{cond}]"
            if term_name not in param_names:
                interim.append(
                    {
                        "comparison": f"Random_vs_{cond}",
                        "estimate_diff_pct_points": "",
                        "std_err": "",
                        "z_value": "",
                        "p_raw": "",
                    }
                )
                continue

            est = _safe_float(params[term_name])
            se = _safe_float(fit.bse[term_name])
            wald = fit.t_test([1.0 if n == term_name else 0.0 for n in param_names])
            z_val = _safe_float(wald.tvalue)
            p_val = _safe_float(wald.pvalue)

            p_vals.append(p_val)
            interim.append(
                {
                    "comparison": f"Random_vs_{cond}",
                    "estimate_diff_pct_points": est,
                    "std_err": se,
                    "z_value": z_val,
                    "p_raw": p_val,
                }
            )

        p_adj = holm_adjust([p for p in p_vals if p != ""])
        p_iter = iter(p_adj)

        for row in interim:
            if row["p_raw"] == "":
                p_holm = ""
                sig = ""
            else:
                p_holm = next(p_iter)
                sig = bool(p_holm < 0.05)

            out_rows.append(
                {
                    "action": action,
                    "comparison": row["comparison"],
                    "model": "GEE Gaussian with exchangeable correlation; cluster=source_file",
                    "global_test": global_test,
                    "global_statistic": global_stat,
                    "global_p_value": global_p,
                    "estimate_diff_pct_points": row["estimate_diff_pct_points"],
                    "std_err": row["std_err"],
                    "z_value": row["z_value"],
                    "p_value_raw": row["p_raw"],
                    "p_value_holm": p_holm,
                    "significant_alpha_0_05": sig,
                    "n_runs_random": int((model_df["condition"] == "Random").sum()),
                    "n_runs_no_ace": int((model_df["condition"] == "No ACE").sum()),
                    "n_runs_moderate_adversity": int(
                        (model_df["condition"] == "Moderate adversity").sum()
                    ),
                    "n_runs_ace": int((model_df["condition"] == "ACE").sum()),
                    "n_clusters_random": model_df.loc[
                        model_df["condition"] == "Random", "source_file"
                    ].nunique(),
                    "n_clusters_no_ace": model_df.loc[
                        model_df["condition"] == "No ACE", "source_file"
                    ].nunique(),
                    "n_clusters_moderate_adversity": model_df.loc[
                        model_df["condition"] == "Moderate adversity", "source_file"
                    ].nunique(),
                    "n_clusters_ace": model_df.loc[
                        model_df["condition"] == "ACE", "source_file"
                    ].nunique(),
                }
            )

    return out_rows


def plot_grouped_bars(data, ci95, output_path, title, actions, conditions):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = list(range(len(actions)))
    width = 0.18
    center_offset = (len(conditions) - 1) / 2

    for i, condition in enumerate(conditions):
        bar_x = [v + (i - center_offset) * width for v in x]
        heights = [data[condition][action] for action in actions]
        errs = [ci95[condition][action] for action in actions]
        ax.bar(
            bar_x,
            heights,
            yerr=errs,
            capsize=3,
            width=width,
            label=condition,
            color=COLORS[condition],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel("Percent of actions (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([ACTION_LABELS[a] for a in actions])
    ax.legend(title="Condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def print_summary_table_figure1a(data):
    print("\nFigure 1A percentages (Fight/Flee/Cry/Chase only):")
    print("Condition              Fight %   Flee %    Cry %    Chase %")
    print("-" * 60)
    for condition in CONDITIONS:
        print(
            f"{condition:<20} "
            f"{data[condition]['fight_count']:>8.2f} "
            f"{data[condition]['flee_count']:>8.2f} "
            f"{data[condition]['cry_count']:>8.2f} "
            f"{data[condition]['chase_count']:>10.2f}"
        )


def print_summary_table_figure1b(data):
    print("\nFigure 1B percentages (non-random only):")
    print("Condition              Learned Helplessness %   Apathy %")
    print("-" * 60)
    for condition in NON_RANDOM_CONDITIONS:
        print(
            f"{condition:<20} "
            f"{data[condition]['learned_helplessness']:>24.2f} "
            f"{data[condition]['apathy']:>11.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build Figure 1 from multiple_runs CSV files using run-level percentages and clustered GEE stats."
        )
    )
    parser.add_argument("--NB", nargs="*", default=[], help="NB CSV files or globs")
    parser.add_argument("--DB", nargs="*", default=[], help="DB CSV files or globs")
    parser.add_argument("--Random", nargs="*", default=[], help="Random CSV files or globs")

    parser.add_argument(
        "--output-a",
        default=os.path.join("FinalProject", "figure1", "figure1A_environment_behavior_percent.png"),
    )
    parser.add_argument(
        "--output-b",
        default=os.path.join("FinalProject", "figure1", "figure1B_helplessness_apathy_percent.png"),
    )
    parser.add_argument(
        "--raw-a-output",
        default=os.path.join("FinalProject", "figure1", "figure1A_raw_percentages.csv"),
    )
    parser.add_argument(
        "--raw-b-output",
        default=os.path.join("FinalProject", "figure1", "figure1B_raw_percentages.csv"),
    )
    parser.add_argument(
        "--stats-output",
        default=os.path.join("FinalProject", "figure1", "figure1A_stats_random_vs_conditions.csv"),
    )

    args = parser.parse_args()

    nb_files = flatten_globs(args.NB)
    db_files = flatten_globs(args.DB)
    random_files = flatten_globs(args.Random)

    if not nb_files and not db_files and not random_files:
        parser.error("No input files found. Provide at least one file via --NB, --DB, or --Random.")

    rows_1a, rows_1b, skipped = extract_figure1_rows(nb_files, db_files, random_files)

    figure1a_data, figure1a_ci95 = aggregate_rows_to_means_and_ci95(rows_1a, ACTIONS, CONDITIONS)
    figure1b_data, figure1b_ci95 = aggregate_rows_to_means_and_ci95(
        rows_1b, FIG1B_ACTIONS, NON_RANDOM_CONDITIONS
    )
    stats_rows = run_figure1a_clustered_stats(rows_1a)

    write_csv(
        args.raw_a_output,
        ["source_file", "source_path", "agent_type", "condition", "run"] + ACTIONS,
        rows_1a,
    )
    write_csv(
        args.raw_b_output,
        ["source_file", "source_path", "agent_type", "condition", "run"] + FIG1B_ACTIONS,
        rows_1b,
    )
    write_csv(
        args.stats_output,
        [
            "action",
            "comparison",
            "model",
            "global_test",
            "global_statistic",
            "global_p_value",
            "estimate_diff_pct_points",
            "std_err",
            "z_value",
            "p_value_raw",
            "p_value_holm",
            "significant_alpha_0_05",
            "n_runs_random",
            "n_runs_no_ace",
            "n_runs_moderate_adversity",
            "n_runs_ace",
            "n_clusters_random",
            "n_clusters_no_ace",
            "n_clusters_moderate_adversity",
            "n_clusters_ace",
        ],
        stats_rows,
    )

    if skipped:
        print("\nSkipped files:")
        for path, reason in skipped:
            print(f"- {path}: {reason}")

    print_summary_table_figure1a(figure1a_data)
    print_summary_table_figure1b(figure1b_data)

    plot_grouped_bars(
        data=figure1a_data,
        ci95=figure1a_ci95,
        output_path=args.output_a,
        title="Figure 1A: Effect of Environment on Behavior (Action Percent)",
        actions=ACTIONS,
        conditions=CONDITIONS,
    )
    plot_grouped_bars(
        data=figure1b_data,
        ci95=figure1b_ci95,
        output_path=args.output_b,
        title="Figure 1B: Learned Helplessness and Apathy by Environment",
        actions=FIG1B_ACTIONS,
        conditions=NON_RANDOM_CONDITIONS,
    )

    print(f"\nSaved Figure 1A to: {args.output_a}")
    print(f"Saved Figure 1B to: {args.output_b}")
    print(f"Saved Figure 1A raw data to: {args.raw_a_output}")
    print(f"Saved Figure 1B raw data to: {args.raw_b_output}")
    print(f"Saved Figure 1A clustered stats to: {args.stats_output}")


if __name__ == "__main__":
    main()
