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

RUN_COLUMNS = ["fight_count", "flee_count", "befriend_count", "chase_count", "cry_count"]


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


def collect_role_rows(files, role_name):
    per_condition = {c: {a: [] for a in ACTIONS} for c in CONDITIONS}
    raw_rows = []
    skipped = []

    for csv_path in files:
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
            row = {
                "source_file": os.path.basename(csv_path),
                "source_path": csv_path,
                "role": role_name,
                "condition": condition,
                "run": run_id,
            }
            for action in ACTIONS:
                pct = safe_percent(values[action], action_total)
                per_condition[condition][action].append(pct)
                row[action] = pct
            raw_rows.append(row)

    mean_data = {
        condition: {action: mean(per_condition[condition][action]) for action in ACTIONS}
        for condition in CONDITIONS
    }
    ci95_data = {
        condition: {
            action: ci95_half_width(per_condition[condition][action]) for action in ACTIONS
        }
        for condition in CONDITIONS
    }
    return mean_data, ci95_data, raw_rows, skipped


def collect_random_baseline(random_files):
    baseline_values = {a: [] for a in ACTIONS}
    skipped = []

    for csv_path in random_files:
        run_data = parse_run_rows(csv_path)
        if not run_data:
            skipped.append((csv_path, "Missing or invalid Run table"))
            continue

        for _, values in run_data:
            action_total = sum(values[a] for a in ACTIONS)
            for action in ACTIONS:
                baseline_values[action].append(safe_percent(values[action], action_total))

    return {action: mean(baseline_values[action]) for action in ACTIONS}, skipped


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


def run_figure2_clustered_stats(raw_rows):
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

        role_terms = [n for n in param_names if n.startswith("C(role") and ":" not in n]
        cond_terms = [n for n in param_names if n.startswith("C(condition") and ":" not in n]
        int_terms = [n for n in param_names if ":" in n]

        role_stat, role_p = _wald_subset(fit, param_names, role_terms)
        cond_stat, cond_p = _wald_subset(fit, param_names, cond_terms)
        int_stat, int_p = _wald_subset(fit, param_names, int_terms)

        base_role = "C(role, Treatment(reference='DB'))[T.NB]"
        int_mod = (
            "C(role, Treatment(reference='DB'))[T.NB]:"
            "C(condition, Treatment(reference='No ACE'))[T.Moderate adversity]"
        )
        int_ace = (
            "C(role, Treatment(reference='DB'))[T.NB]:"
            "C(condition, Treatment(reference='No ACE'))[T.ACE]"
        )

        comparisons = []
        for condition, weights in [
            ("No ACE", {base_role: 1.0}),
            ("Moderate adversity", {base_role: 1.0, int_mod: 1.0}),
            ("ACE", {base_role: 1.0, int_ace: 1.0}),
        ]:
            est, se, z_val, p_raw = _contrast_from_terms(fit, param_names, weights)
            comparisons.append((condition, est, se, z_val, p_raw))

        p_adj = holm_adjust([c[4] for c in comparisons if c[4] != ""])
        p_iter = iter(p_adj)

        for condition, est, se, z_val, p_raw in comparisons:
            if p_raw == "":
                p_holm = ""
                sig = ""
            else:
                p_holm = next(p_iter)
                sig = bool(p_holm < 0.05)

            out_rows.append(
                {
                    "analysis_type": "DB_vs_NB_within_condition_clustered",
                    "action": action,
                    "comparison": f"DB_vs_NB_at_{condition}",
                    "model": "GEE Gaussian exchangeable; cluster=source_file",
                    "estimate_diff_pct_points": est,
                    "std_err": se,
                    "z_value": z_val,
                    "p_value_raw": p_raw,
                    "p_value_holm": p_holm,
                    "significant_alpha_0_05": sig,
                    "role_main_wald_stat": role_stat,
                    "role_main_wald_p": role_p,
                    "condition_main_wald_stat": cond_stat,
                    "condition_main_wald_p": cond_p,
                    "interaction_wald_stat": int_stat,
                    "interaction_wald_p": int_p,
                    "n_runs": len(model_df),
                    "n_clusters": model_df["source_file"].nunique(),
                }
            )

        cond_mod = "C(condition, Treatment(reference='No ACE'))[T.Moderate adversity]"
        cond_ace = "C(condition, Treatment(reference='No ACE'))[T.ACE]"
        role_condition_rows = []

        for role in ROLES:
            for comp_name, weights in [
                ("No_ACE_vs_Moderate_adversity", {cond_mod: 1.0}),
                ("No_ACE_vs_ACE", {cond_ace: 1.0}),
                ("Moderate_adversity_vs_ACE", {cond_ace: 1.0, cond_mod: -1.0}),
            ]:
                if role == "NB":
                    if comp_name == "No_ACE_vs_Moderate_adversity":
                        weights[int_mod] = weights.get(int_mod, 0.0) + 1.0
                    elif comp_name == "No_ACE_vs_ACE":
                        weights[int_ace] = weights.get(int_ace, 0.0) + 1.0
                    elif comp_name == "Moderate_adversity_vs_ACE":
                        weights[int_ace] = weights.get(int_ace, 0.0) + 1.0
                        weights[int_mod] = weights.get(int_mod, 0.0) - 1.0

                est, se, z_val, p_raw = _contrast_from_terms(fit, param_names, weights)
                role_condition_rows.append((role, comp_name, est, se, z_val, p_raw))

        for role in ROLES:
            role_slice = [r for r in role_condition_rows if r[0] == role]
            adj = holm_adjust([r[5] for r in role_slice if r[5] != ""])
            adj_iter = iter(adj)
            for _, comp_name, est, se, z_val, p_raw in role_slice:
                if p_raw == "":
                    p_holm = ""
                    sig = ""
                else:
                    p_holm = next(adj_iter)
                    sig = bool(p_holm < 0.05)

                out_rows.append(
                    {
                        "analysis_type": "Within_role_across_conditions_clustered",
                        "action": action,
                        "comparison": f"{role}_{comp_name}",
                        "model": "GEE Gaussian exchangeable; cluster=source_file",
                        "estimate_diff_pct_points": est,
                        "std_err": se,
                        "z_value": z_val,
                        "p_value_raw": p_raw,
                        "p_value_holm": p_holm,
                        "significant_alpha_0_05": sig,
                        "role_main_wald_stat": role_stat,
                        "role_main_wald_p": role_p,
                        "condition_main_wald_stat": cond_stat,
                        "condition_main_wald_p": cond_p,
                        "interaction_wald_stat": int_stat,
                        "interaction_wald_p": int_p,
                        "n_runs": len(model_df),
                        "n_clusters": model_df["source_file"].nunique(),
                    }
                )

    return out_rows


def run_figure2_comprehensive_stats(db_rows, nb_rows, random_rows):
    """Comprehensive analysis including Random baseline comparisons for each role×condition."""
    out_rows = []
    
    for action in ACTIONS:
        # Build comprehensive dataset with all three roles
        all_rows = []
        for role, rows in [("DB", db_rows), ("NB", nb_rows), ("Random", random_rows)]:
            for row in rows:
                row_copy = row.copy()
                row_copy["role"] = role
                all_rows.append(row_copy)
        
        df = pd.DataFrame(all_rows)
        model_df = df[["role", "condition", "source_file", action]].copy().dropna()
        
        # Use Random as reference for role, No ACE as reference for condition
        formula = (
            f"{action} ~ C(role, Treatment(reference='Random')) "
            "+ C(condition, Treatment(reference='No ACE')) "
            "+ C(role, Treatment(reference='Random')):C(condition, Treatment(reference='No ACE'))"
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
        
        # Extract coefficients
        db_coef = "C(role, Treatment(reference='Random'))[T.DB]"
        nb_coef = "C(role, Treatment(reference='Random'))[T.NB]"
        mod_cond = "C(condition, Treatment(reference='No ACE'))[T.Moderate adversity]"
        ace_cond = "C(condition, Treatment(reference='No ACE'))[T.ACE]"
        db_mod_int = f"{db_coef}:C(condition, Treatment(reference='No ACE'))[T.Moderate adversity]"
        db_ace_int = f"{db_coef}:C(condition, Treatment(reference='No ACE'))[T.ACE]"
        nb_mod_int = f"{nb_coef}:C(condition, Treatment(reference='No ACE'))[T.Moderate adversity]"
        nb_ace_int = f"{nb_coef}:C(condition, Treatment(reference='No ACE'))[T.ACE]"
        
        # Collect contrasts for table output
        contrasts_list = []
        
        # Random vs DB contrasts for each condition
        for condition, weights in [
            ("No ACE", {db_coef: -1.0}),  # DB - Random = -1 * (Random - DB)
            ("Moderate adversity", {db_coef: -1.0, db_mod_int: -1.0}),
            ("ACE", {db_coef: -1.0, db_ace_int: -1.0}),
        ]:
            est, se, z_val, p_raw = _contrast_from_terms(fit, param_names, weights)
            contrasts_list.append({
                "action": action,
                "comparison_type": "Random_vs_DB",
                "role": "DB",
                "condition": condition,
                "estimate_diff": est,
                "std_err": se,
                "z": z_val,
                "p": p_raw,
            })
        
        # Random vs NB contrasts for each condition
        for condition, weights in [
            ("No ACE", {nb_coef: -1.0}),
            ("Moderate adversity", {nb_coef: -1.0, nb_mod_int: -1.0}),
            ("ACE", {nb_coef: -1.0, nb_ace_int: -1.0}),
        ]:
            est, se, z_val, p_raw = _contrast_from_terms(fit, param_names, weights)
            contrasts_list.append({
                "action": action,
                "comparison_type": "Random_vs_NB",
                "role": "NB",
                "condition": condition,
                "estimate_diff": est,
                "std_err": se,
                "z": z_val,
                "p": p_raw,
            })
        
        # DB vs NB contrasts for each condition
        for condition, weights in [
            ("No ACE", {nb_coef: 1.0, db_coef: -1.0}),
            ("Moderate adversity", {nb_coef: 1.0, db_coef: -1.0, nb_mod_int: 1.0, db_mod_int: -1.0}),
            ("ACE", {nb_coef: 1.0, db_coef: -1.0, nb_ace_int: 1.0, db_ace_int: -1.0}),
        ]:
            est, se, z_val, p_raw = _contrast_from_terms(fit, param_names, weights)
            contrasts_list.append({
                "action": action,
                "comparison_type": "DB_vs_NB",
                "role": "both",
                "condition": condition,
                "estimate_diff": est,
                "std_err": se,
                "z": z_val,
                "p": p_raw,
            })
        
        # Within-role condition comparisons
        # DB: Condition comparisons
        for comp_name, weights in [
            ("No_ACE_vs_Moderate_adversity", {mod_cond: -1.0, db_mod_int: -1.0}),
            ("No_ACE_vs_ACE", {ace_cond: -1.0, db_ace_int: -1.0}),
            ("Moderate_adversity_vs_ACE", {ace_cond: -1.0, db_ace_int: -1.0, mod_cond: 1.0, db_mod_int: 1.0}),
        ]:
            est, se, z_val, p_raw = _contrast_from_terms(fit, param_names, weights)
            contrasts_list.append({
                "action": action,
                "comparison_type": "DB_condition",
                "role": "DB",
                "condition": comp_name,
                "estimate_diff": est,
                "std_err": se,
                "z": z_val,
                "p": p_raw,
            })
        
        # NB: Condition comparisons
        for comp_name, weights in [
            ("No_ACE_vs_Moderate_adversity", {mod_cond: -1.0, nb_mod_int: -1.0}),
            ("No_ACE_vs_ACE", {ace_cond: -1.0, nb_ace_int: -1.0}),
            ("Moderate_adversity_vs_ACE", {ace_cond: -1.0, nb_ace_int: -1.0, mod_cond: 1.0, nb_mod_int: 1.0}),
        ]:
            est, se, z_val, p_raw = _contrast_from_terms(fit, param_names, weights)
            contrasts_list.append({
                "action": action,
                "comparison_type": "NB_condition",
                "role": "NB",
                "condition": comp_name,
                "estimate_diff": est,
                "std_err": se,
                "z": z_val,
                "p": p_raw,
            })
        
        # Apply Holm correction per role×comparison_type group
        comparison_groups = {}
        for contrast in contrasts_list:
            key = (contrast["role"], contrast["comparison_type"])
            if key not in comparison_groups:
                comparison_groups[key] = []
            comparison_groups[key].append(contrast)
        
        # Apply Holm correction and add to output
        for group_contrasts in comparison_groups.values():
            p_vals = [c["p"] for c in group_contrasts if c["p"] != ""]
            p_adj = holm_adjust(p_vals)
            p_iter = iter(p_adj)
            
            for contrast in group_contrasts:
                if contrast["p"] == "":
                    p_holm = ""
                    significant = ""
                else:
                    p_holm = next(p_iter)
                    significant = bool(p_holm < 0.05)
                
                n_runs = len(model_df)
                n_clusters = model_df["source_file"].nunique()
                
                out_rows.append({
                    "action": contrast["action"],
                    "comparison": contrast["comparison_type"],
                    "role": contrast["role"],
                    "condition": contrast["condition"],
                    "estimate_diff": contrast["estimate_diff"],
                    "std_err": contrast["std_err"],
                    "z": contrast["z"],
                    "p_value": contrast["p"],
                    "p_holm": p_holm,
                    "significant": significant,
                    "n_runs": n_runs,
                    "n_clusters": n_clusters,
                })
    
    return out_rows


def plot_panel(ax, panel_title, data, ci95_data, random_baseline=None):
    x = list(range(len(ACTIONS)))
    width = 0.24
    center_offset = (len(CONDITIONS) - 1) / 2

    for i, condition in enumerate(CONDITIONS):
        bar_x = [v + (i - center_offset) * width for v in x]
        heights = [data[condition][action] for action in ACTIONS]
        errs = [ci95_data[condition][action] for action in ACTIONS]
        ax.bar(
            bar_x,
            heights,
            yerr=errs,
            capsize=3,
            width=width,
            color=CONDITION_COLORS[condition],
            edgecolor="black",
            linewidth=0.5,
            label=condition,
        )

    if random_baseline is not None:
        baseline_heights = [random_baseline[action] for action in ACTIONS]
        ax.plot(
            x,
            baseline_heights,
            color="#7a7a7a",
            linewidth=1.5,
            linestyle="--",
            marker="o",
            markersize=4,
            alpha=0.6,
            label="Random baseline",
        )

    ax.set_title(panel_title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([ACTION_LABELS[action] for action in ACTIONS])
    ax.set_ylabel("Percent of simple actions (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False)


def save_single_panel_figure(panel_title, data, ci95_data, output_path, random_baseline=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_panel(ax, panel_title, data, ci95_data, random_baseline=random_baseline)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(role_name, data):
    print(f"\n{role_name} percentages (% within Fight/Flee/Cry/Chase):")
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


def main():
    parser = argparse.ArgumentParser(
        description="Figure 2 with run-level clustered GEE stats: action ~ role + condition + role*condition"
    )
    parser.add_argument("--DB", nargs="*", default=[], help="DB CSV files or globs")
    parser.add_argument("--NB", nargs="*", default=[], help="NB CSV files or globs")
    parser.add_argument("--Random", nargs="*", default=[], help="Random CSV files or globs")
    parser.add_argument(
        "--show-random-baseline",
        action="store_true",
        help="Overlay a faint random baseline line on both panels",
    )
    parser.add_argument(
        "--output-a",
        default=os.path.join("FinalProject", "figure2", "figure2A_db_role_differences.png"),
    )
    parser.add_argument(
        "--output-b",
        default=os.path.join("FinalProject", "figure2", "figure2B_nb_role_differences.png"),
    )
    parser.add_argument(
        "--raw-output",
        default=os.path.join("FinalProject", "figure2", "figure2_raw_percentages.csv"),
    )
    parser.add_argument(
        "--stats-output",
        default=os.path.join("FinalProject", "figure2", "figure2_stats.csv"),
    )
    parser.add_argument(
        "--comprehensive-stats-output",
        default=os.path.join("FinalProject", "figure2", "figure2_comprehensive_stats.csv"),
        help="Comprehensive table including Random baseline comparisons for each role×condition",
    )

    args = parser.parse_args()

    db_files = flatten_globs(args.DB)
    nb_files = flatten_globs(args.NB)
    random_files = flatten_globs(args.Random)

    if not db_files:
        parser.error("No DB files found. Provide one or more files with --DB.")
    if not nb_files:
        parser.error("No NB files found. Provide one or more files with --NB.")

    db_data, db_ci95_data, db_rows, db_skipped = collect_role_rows(db_files, "DB")
    nb_data, nb_ci95_data, nb_rows, nb_skipped = collect_role_rows(nb_files, "NB")
    raw_rows = db_rows + nb_rows

    random_baseline = None
    random_skipped = []
    random_rows = []
    if args.show_random_baseline and random_files:
        random_baseline, random_skipped = collect_random_baseline(random_files)
    
    # Also collect Random rows for comprehensive stats if provided
    if random_files:
        _, _, random_rows, random_skipped_comprehensive = collect_role_rows(random_files, "Random")

    all_skipped = db_skipped + nb_skipped + random_skipped
    if all_skipped:
        print("\nSkipped files:")
        for path, reason in all_skipped:
            print(f"- {path}: {reason}")

    stats_rows = run_figure2_clustered_stats(raw_rows)
    
    # Generate comprehensive statistics including Random baseline comparisons
    comprehensive_stats_rows = run_figure2_comprehensive_stats(db_rows, nb_rows, random_rows)

    write_csv(
        args.raw_output,
        ["source_file", "source_path", "role", "condition", "run"] + ACTIONS,
        raw_rows,
    )
    write_csv(
        args.stats_output,
        [
            "analysis_type",
            "action",
            "comparison",
            "model",
            "estimate_diff_pct_points",
            "std_err",
            "z_value",
            "p_value_raw",
            "p_value_holm",
            "significant_alpha_0_05",
            "role_main_wald_stat",
            "role_main_wald_p",
            "condition_main_wald_stat",
            "condition_main_wald_p",
            "interaction_wald_stat",
            "interaction_wald_p",
            "n_runs",
            "n_clusters",
        ],
        stats_rows,
    )
    
    write_csv(
        args.comprehensive_stats_output,
        [
            "action",
            "comparison",
            "role",
            "condition",
            "estimate_diff",
            "std_err",
            "z",
            "p_value",
            "p_holm",
            "significant",
            "n_runs",
            "n_clusters",
        ],
        comprehensive_stats_rows,
    )

    print_summary("Figure 2A (DB)", db_data)
    print_summary("Figure 2B (NB)", nb_data)

    save_single_panel_figure(
        "Figure 2A: DB Agents",
        db_data,
        db_ci95_data,
        args.output_a,
        random_baseline=random_baseline,
    )
    save_single_panel_figure(
        "Figure 2B: NB Agents",
        nb_data,
        nb_ci95_data,
        args.output_b,
        random_baseline=random_baseline,
    )

    print(f"\nSaved Figure 2A to: {args.output_a}")
    print(f"Saved Figure 2B to: {args.output_b}")
    print(f"Saved Figure 2 raw data to: {args.raw_output}")
    print(f"Saved Figure 2 clustered stats to: {args.stats_output}")
    print(f"Saved Figure 2 comprehensive stats (Random vs roles) to: {args.comprehensive_stats_output}")


if __name__ == "__main__":
    main()
