"""
figure2v2Analysis.py - Statistical analysis for loss, reward, and behavioral entropy.

This script reads the multiple-runs series CSV files, groups them by agent type
(DB vs NB) and adversity condition (LowThreat, ModerateThreat, HighThreat), and
computes late-training statistics for:

- Loss convergence: slope of the final training window
- Reward convergence: slope of the final training window
- Behavioral entropy stability: slope of the final training window
- Loss vs entropy relationship: Spearman correlation in the late window

The script reports mean slope, standard deviation, SEM, 95% CI, and a one-sample
 t-test against zero for each slope-based metric. It also writes a CSV summary.

Usage:
    python figure2v2Analysis.py
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from situations import Action

FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "Figures")
OUTPUT_DIR = FIGURES_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITIONS = ["LowThreat", "ModerateThreat", "HighThreat"]
AGENT_TYPES = ["DB", "NB"]
DEFAULT_ROLLING_WINDOW = 50
DEFAULT_LATE_WINDOW = 50


@dataclass
class GroupData:
    loss: np.ndarray
    reward: np.ndarray
    action: np.ndarray
    steps: np.ndarray


def find_series_csv_files(workspace_root: str) -> Dict[str, List[str]]:
    """Return series CSV files grouped by agent type."""
    db_files = glob.glob(os.path.join(workspace_root, "multiple_runs_series_DB_*.csv"))
    nb_files = glob.glob(os.path.join(workspace_root, "multiple_runs_series_NB_*.csv"))

    db_files = [path for path in db_files if "Archive" not in path]
    nb_files = [path for path in nb_files if "Archive" not in path]

    return {"DB": sorted(db_files), "NB": sorted(nb_files)}


def detect_condition(csv_file: str) -> Optional[str]:
    """Infer the adversity condition from the filename."""
    filename = os.path.basename(csv_file)
    for condition in CONDITIONS:
        if condition in filename:
            return condition
    return None


def extract_run_numbers(df: pd.DataFrame) -> List[int]:
    """Extract sorted run indices from runN_* column names."""
    run_numbers = set()
    for column in df.columns:
        if column.startswith("run") and "_" in column:
            run_number = int(column.split("_")[0][3:])
            run_numbers.add(run_number)
    return sorted(run_numbers)


def build_combined_matrices(df: pd.DataFrame, agent_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build combined loss, reward, and action matrices for one CSV file."""
    run_numbers = extract_run_numbers(df)
    step_count = len(df)
    run_count = len(run_numbers)

    loss_array = np.zeros((step_count, run_count), dtype=float)
    reward_array = np.zeros((step_count, run_count), dtype=float)
    action_array = np.zeros((run_count, step_count), dtype=float)

    for run_index, run_number in enumerate(run_numbers):
        l_loss_col = f"run{run_number}_l_loss"
        l_reward_col = f"run{run_number}_l_reward"
        action_col = f"run{run_number}_action"

        if agent_type == "DB":
            loss_col = f"run{run_number}_db_loss"
            reward_col = f"run{run_number}_db_reward"
        else:
            loss_col = f"run{run_number}_nb_loss"
            reward_col = f"run{run_number}_nb_reward"

        loss_array[:, run_index] = df[l_loss_col].to_numpy(dtype=float) + df[loss_col].to_numpy(dtype=float)
        reward_array[:, run_index] = df[l_reward_col].to_numpy(dtype=float) + df[reward_col].to_numpy(dtype=float)
        action_array[run_index, :] = df[action_col].to_numpy(dtype=float)

    return loss_array, reward_array, action_array


def load_condition_data(csv_files: List[str], agent_type: str) -> Dict[str, GroupData]:
    """Load and concatenate all runs for each condition."""
    grouped_loss: Dict[str, List[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    grouped_reward: Dict[str, List[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    grouped_action: Dict[str, List[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    steps = None

    for csv_file in csv_files:
        condition = detect_condition(csv_file)
        if condition is None:
            print(f"  Skipping {os.path.basename(csv_file)} (no condition detected)")
            continue

        df = pd.read_csv(csv_file)
        if steps is None:
            steps = df["step"].to_numpy()

        loss_array, reward_array, action_array = build_combined_matrices(df, agent_type)
        grouped_loss[condition].append(loss_array)
        grouped_reward[condition].append(reward_array)
        grouped_action[condition].append(action_array)
        print(
            f"  Loaded {os.path.basename(csv_file)} -> {condition}: "
            f"{loss_array.shape[1]} runs"
        )

    if steps is None:
        return {}

    condition_data: Dict[str, GroupData] = {}
    for condition in CONDITIONS:
        if grouped_loss[condition]:
            condition_data[condition] = GroupData(
                loss=np.concatenate(grouped_loss[condition], axis=1),
                reward=np.concatenate(grouped_reward[condition], axis=1),
                action=np.concatenate(grouped_action[condition], axis=0),
                steps=steps,
            )
        else:
            condition_data[condition] = GroupData(
                loss=np.array([]),
                reward=np.array([]),
                action=np.array([]),
                steps=steps,
            )

    return condition_data


def trailing_mean(series: np.ndarray, window: int) -> np.ndarray:
    """Compute a trailing mean for a 1D series."""
    if series.size == 0:
        return series

    smoothed = np.zeros_like(series, dtype=float)
    for index in range(len(series)):
        start = max(0, index - window + 1)
        values = series[start : index + 1]
        finite_values = values[np.isfinite(values)]
        smoothed[index] = float(np.mean(finite_values)) if finite_values.size else 0.0
    return smoothed


def rolling_entropy_for_run(actions: np.ndarray, window: int, n_actions: int) -> np.ndarray:
    """Compute trailing behavioral entropy for a single run."""
    entropy = np.zeros(actions.size, dtype=float)
    for step in range(actions.size):
        start = max(0, step - window + 1)
        window_actions = actions[start : step + 1]
        window_actions = window_actions[np.isfinite(window_actions)]
        window_actions = window_actions[window_actions >= 0]

        if window_actions.size == 0:
            entropy[step] = 0.0
            continue

        counts = np.bincount(window_actions.astype(int), minlength=n_actions)
        total = counts.sum()
        if total == 0:
            entropy[step] = 0.0
            continue

        proportions = counts / total
        proportions = proportions[proportions > 0]
        entropy[step] = -np.sum(proportions * np.log(proportions))

    return entropy


def compute_per_run_curves(group_data: GroupData, rolling_window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-run trailing curves for loss, reward, and entropy."""
    if group_data.loss.size == 0:
        return np.array([]), np.array([]), np.array([])

    n_steps, n_runs = group_data.loss.shape
    n_actions = len(Action)

    loss_curves = np.zeros((n_runs, n_steps), dtype=float)
    reward_curves = np.zeros((n_runs, n_steps), dtype=float)
    entropy_curves = np.zeros((n_runs, n_steps), dtype=float)

    for run_index in range(n_runs):
        loss_curves[run_index, :] = trailing_mean(group_data.loss[:, run_index], rolling_window)
        reward_curves[run_index, :] = trailing_mean(group_data.reward[:, run_index], rolling_window)
        entropy_curves[run_index, :] = rolling_entropy_for_run(group_data.action[run_index, :], rolling_window, n_actions)

    return loss_curves, reward_curves, entropy_curves


def slope_over_window(series: np.ndarray, window: int) -> float:
    """Fit a simple linear trend over the final window and return the slope."""
    finite_series = series[np.isfinite(series)]
    if finite_series.size < 2:
        return 0.0

    segment = finite_series[-window:] if finite_series.size >= window else finite_series
    if segment.size < 2:
        return 0.0

    x_values = np.arange(segment.size, dtype=float)
    slope = np.polyfit(x_values, segment, deg=1)[0]
    return float(slope)


def compute_relative_change(values: np.ndarray) -> Tuple[float, float, float]:
    """
    How much did the metric change from start to end as a percentage of its initial value.
    Uses first 200 steps and last 200 steps of the provided 1D series.
    Returns (early_mean, late_mean, percent_change)
    """
    if values is None or values.size == 0:
        return 0.0, 0.0, 0.0

    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return 0.0, 0.0, 0.0

    early_segment = finite_vals[:200] if finite_vals.size >= 200 else finite_vals
    late_segment = finite_vals[-200:] if finite_vals.size >= 200 else finite_vals

    early_mean = float(np.mean(early_segment)) if early_segment.size else 0.0
    late_mean = float(np.mean(late_segment)) if late_segment.size else 0.0

    if early_mean == 0.0:
        pct_change = 0.0
    else:
        pct_change = float((late_mean - early_mean) / abs(early_mean) * 100.0)

    return early_mean, late_mean, pct_change


def late_window_mean(series: np.ndarray, window: int) -> float:
    """Mean of the final window of a series."""
    finite_series = series[np.isfinite(series)]
    if finite_series.size == 0:
        return 0.0
    segment = finite_series[-window:] if finite_series.size >= window else finite_series
    return float(np.mean(segment))


def summarize_slopes(slopes: np.ndarray) -> Dict[str, float]:
    """Summarize a vector of per-run slopes."""
    slopes = slopes[np.isfinite(slopes)]
    n_runs = int(slopes.size)
    if n_runs == 0:
        return {
            "n_runs": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
        }

    mean_value = float(np.mean(slopes))
    std_value = float(np.std(slopes, ddof=1)) if n_runs > 1 else 0.0
    sem_value = float(std_value / np.sqrt(n_runs)) if n_runs > 0 else 0.0

    if n_runs > 1:
        t_stat, p_value = stats.ttest_1samp(slopes, popmean=0.0, nan_policy="omit")
        t_stat = float(t_stat)
        p_value = float(p_value)
        ci_half_width = float(stats.t.ppf(0.975, df=n_runs - 1) * sem_value)
    else:
        t_stat = 0.0
        p_value = 1.0
        ci_half_width = 0.0

    return {
        "n_runs": n_runs,
        "mean": mean_value,
        "std": std_value,
        "sem": sem_value,
        "ci95_low": mean_value - ci_half_width,
        "ci95_high": mean_value + ci_half_width,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def analyze_group(group_data: GroupData, rolling_window: int, late_window: int) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Analyze one condition group and return metric summaries, correlations, and relative-change info."""
    loss_curves, reward_curves, entropy_curves = compute_per_run_curves(group_data, rolling_window)
    if loss_curves.size == 0:
        return {}, {}, {}

    metric_slopes = {}
    metric_late_means = {}

    for metric_name, curves in (
        ("loss", loss_curves),
        ("reward", reward_curves),
        ("entropy", entropy_curves),
    ):
        per_run_slopes = np.array([slope_over_window(curve, late_window) for curve in curves], dtype=float)
        metric_slopes[metric_name] = summarize_slopes(per_run_slopes)

        per_run_late_means = np.array([late_window_mean(curve, late_window) for curve in curves], dtype=float)
        metric_late_means[metric_name] = {
            "values": per_run_late_means,
            "mean": float(np.mean(per_run_late_means)) if per_run_late_means.size else 0.0,
            "std": float(np.std(per_run_late_means, ddof=1)) if per_run_late_means.size > 1 else 0.0,
        }

    loss_entropy_corr = {"rho": 0.0, "p_value": 1.0, "n_runs": 0}
    loss_values = metric_late_means["loss"]["values"]
    entropy_values = metric_late_means["entropy"]["values"]
    valid_mask = np.isfinite(loss_values) & np.isfinite(entropy_values)
    if np.sum(valid_mask) > 1:
        rho, p_value = stats.spearmanr(loss_values[valid_mask], entropy_values[valid_mask])
        loss_entropy_corr = {
            "rho": float(rho),
            "p_value": float(p_value),
            "n_runs": int(np.sum(valid_mask)),
        }

    # Compute mean across runs (per-step) and relative change for loss and entropy
    metric_changes: Dict[str, Dict[str, float]] = {}
    try:
        loss_mean_series = np.nanmean(loss_curves, axis=0) if loss_curves.size else np.array([])
    except Exception:
        loss_mean_series = np.array([])

    try:
        entropy_mean_series = np.nanmean(entropy_curves, axis=0) if entropy_curves.size else np.array([])
    except Exception:
        entropy_mean_series = np.array([])

    loss_early, loss_late, loss_pct = compute_relative_change(loss_mean_series)
    ent_early, ent_late, ent_pct = compute_relative_change(entropy_mean_series)

    metric_changes["loss"] = {
        "early_mean": loss_early,
        "late_mean": loss_late,
        "pct_change": loss_pct,
    }
    metric_changes["entropy"] = {
        "early_mean": ent_early,
        "late_mean": ent_late,
        "pct_change": ent_pct,
    }

    return metric_slopes, loss_entropy_corr, metric_changes
    

    # unreachable



def build_summary_table(agent_groups: Dict[str, Dict[str, GroupData]], rolling_window: int, late_window: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build summary and correlation tables for all agent types and conditions."""
    summary_rows = []
    correlation_rows = []
    change_rows = []

    for agent_type in AGENT_TYPES:
        for condition in CONDITIONS:
            group_data = agent_groups.get(agent_type, {}).get(condition)
            if group_data is None or group_data.loss.size == 0:
                continue

            metric_slopes, loss_entropy_corr, metric_changes = analyze_group(group_data, rolling_window, late_window)

            for metric_name, summary in metric_slopes.items():
                summary_rows.append(
                    {
                        "agent_type": agent_type,
                        "condition": condition,
                        "metric": metric_name,
                        "n_runs": summary["n_runs"],
                        "mean_slope": summary["mean"],
                        "std_slope": summary["std"],
                        "sem_slope": summary["sem"],
                        "ci95_low": summary["ci95_low"],
                        "ci95_high": summary["ci95_high"],
                        "t_stat": summary["t_stat"],
                        "p_value": summary["p_value"],
                    }
                )

            correlation_rows.append(
                {
                    "agent_type": agent_type,
                    "condition": condition,
                    "relationship": "late_window_loss_entropy_spearman",
                    "n_runs": loss_entropy_corr["n_runs"],
                    "spearman_rho": loss_entropy_corr["rho"],
                    "p_value": loss_entropy_corr["p_value"],
                }
            )

            # record relative-change results for loss and entropy
            if metric_changes:
                for m in ("loss", "entropy"):
                    ch = metric_changes.get(m, None)
                    if ch is not None:
                        change_rows.append(
                            {
                                "agent_type": agent_type,
                                "condition": condition,
                                "metric": m,
                                "early_mean": ch["early_mean"],
                                "late_mean": ch["late_mean"],
                                "pct_change": ch["pct_change"],
                            }
                        )

    return pd.DataFrame(summary_rows), pd.DataFrame(correlation_rows), pd.DataFrame(change_rows)


def print_summary(summary_df: pd.DataFrame, correlation_df: pd.DataFrame) -> None:
    """Print a compact report to the console."""
    print("\n" + "=" * 80)
    print("Late-window slope summary")
    print("=" * 80)
    if summary_df.empty:
        print("No data found.")
        return

    for _, row in summary_df.iterrows():
        print(
            f"{row['agent_type']:>2} | {row['condition']:<15} | {row['metric']:<7} | "
            f"mean slope={row['mean_slope']:+.6f} | 95% CI [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | "
            f"t={row['t_stat']:+.3f}, p={row['p_value']:.4g}, n={int(row['n_runs'])}"
        )

    print("\n" + "=" * 80)
    print("Late-window loss-entropy correlation")
    print("=" * 80)
    if correlation_df.empty:
        print("No correlation data found.")
        return

    for _, row in correlation_df.iterrows():
        print(
            f"{row['agent_type']:>2} | {row['condition']:<15} | "
            f"Spearman rho={row['spearman_rho']:+.4f}, p={row['p_value']:.4g}, n={int(row['n_runs'])}"
        )


def main() -> None:
    print("=" * 80)
    print("Figure 2v2 Analysis")
    print("=" * 80)

    csv_groups = find_series_csv_files(WORKSPACE_ROOT)
    print(f"Found {len(csv_groups['DB'])} DB CSV files and {len(csv_groups['NB'])} NB CSV files")

    if not csv_groups["DB"] and not csv_groups["NB"]:
        print("ERROR: No series CSV files found.")
        return

    agent_groups: Dict[str, Dict[str, GroupData]] = {"DB": {}, "NB": {}}
    for agent_type, csv_files in csv_groups.items():
        print(f"\nProcessing {agent_type} files...")
        agent_groups[agent_type] = load_condition_data(csv_files, agent_type)

    summary_df, correlation_df, change_df = build_summary_table(
        agent_groups=agent_groups,
        rolling_window=DEFAULT_ROLLING_WINDOW,
        late_window=DEFAULT_LATE_WINDOW,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(OUTPUT_DIR, f"figure2v2Analysis_summary_{timestamp}.csv")
    correlation_path = os.path.join(OUTPUT_DIR, f"figure2v2Analysis_correlation_{timestamp}.csv")
    change_path = os.path.join(OUTPUT_DIR, f"figure2v2Analysis_relative_change_{timestamp}.csv")

    summary_df.to_csv(summary_path, index=False)
    correlation_df.to_csv(correlation_path, index=False)
    change_df.to_csv(change_path, index=False)

    print_summary(summary_df, correlation_df)

    print("\nSaved summary to:")
    print(f"  {summary_path}")
    print(f"  {correlation_path}")
    print(f"  {change_path}")

    # Print relative-change summary for quick inspection
    if not change_df.empty:
        print("\n" + "=" * 80)
        print("Relative change (early vs late) — percent of initial value")
        print("=" * 80)
        for _, row in change_df.iterrows():
            print(
                f"{row['agent_type']:>2} | {row['condition']:<15} | {row['metric']:<7} | "
                f"early={row['early_mean']:+.4f}, late={row['late_mean']:+.4f}, pct_change={row['pct_change']:+.1f}%"
            )
    print("\nDone.")


if __name__ == "__main__":
    main()
