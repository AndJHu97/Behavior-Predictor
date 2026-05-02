"""
figure2v2BehavioralEntropy.py - Plot behavioral entropy across training steps.

This script reads the multiple-runs series CSV files, extracts the per-step
run action columns, computes Shannon behavioral entropy across runs for each
training step, and plots the entropy trajectories for LowThreat, ModerateThreat,
and HighThreat conditions separately for DB and NB agents.

Usage:
    python figure2v2BehavioralEntropy.py
"""

import glob
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from situations import Action

FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "Figures")
OUTPUT_DIR = FIGURES_DIR

CONDITIONS = ["LowThreat", "ModerateThreat", "HighThreat"]
CONDITION_LABELS = {
    "LowThreat": "Low ACE",
    "ModerateThreat": "Moderate adversity",
    "HighThreat": "High ACE",
}
CONDITION_COLORS = {
    "LowThreat": "#2a9d8f",
    "ModerateThreat": "#e9c46a",
    "HighThreat": "#e76f51",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_series_csv_files(workspace_root):
    """Return DB and NB series CSV files grouped by condition."""
    db_files = glob.glob(os.path.join(workspace_root, "multiple_runs_series_DB_*.csv"))
    nb_files = glob.glob(os.path.join(workspace_root, "multiple_runs_series_NB_*.csv"))

    db_files = [path for path in db_files if "Archive" not in path]
    nb_files = [path for path in nb_files if "Archive" not in path]

    return sorted(db_files), sorted(nb_files)


def detect_condition(csv_file):
    """Infer the adversity condition from the filename."""
    filename = os.path.basename(csv_file)
    for condition in CONDITIONS:
        if condition in filename:
            return condition
    return None


def extract_action_matrix(df):
    """Extract a [runs, steps] action matrix from runN_action columns."""
    action_columns = [column for column in df.columns if column.startswith("run") and column.endswith("_action")]
    if not action_columns:
        raise ValueError("No run*_action columns found in CSV")

    run_numbers = sorted({int(column.split("_")[0][3:]) for column in action_columns})
    step_count = len(df)

    action_matrix = np.zeros((len(run_numbers), step_count), dtype=float)
    for index, run_number in enumerate(run_numbers):
        column_name = f"run{run_number}_action"
        if column_name not in df.columns:
            raise ValueError(f"Missing action column: {column_name}")
        action_matrix[index, :] = df[column_name].to_numpy()

    return action_matrix


def load_condition_action_data(csv_files):
    """Load and concatenate action matrices for each condition."""
    condition_runs = {condition: [] for condition in CONDITIONS}
    step_values = None

    for csv_file in csv_files:
        condition = detect_condition(csv_file)
        if condition is None:
            print(f"  Skipping {os.path.basename(csv_file)} (no condition detected)")
            continue

        df = pd.read_csv(csv_file)
        if step_values is None:
            step_values = df["step"].to_numpy()

        action_matrix = extract_action_matrix(df)
        condition_runs[condition].append(action_matrix)
        print(f"  Loaded {os.path.basename(csv_file)} -> {condition}: {action_matrix.shape[0]} runs")

    condition_data = {}
    for condition, matrices in condition_runs.items():
        if matrices:
            condition_data[condition] = np.concatenate(matrices, axis=0)
        else:
            condition_data[condition] = None

    return step_values, condition_data


def compute_behavioral_entropy(action_matrix, n_actions=None, window=50):
    """Compute mean entropy across runs and run-to-run std at each step."""
    if action_matrix is None or action_matrix.size == 0:
        return None, None, None

    n_runs, n_steps = action_matrix.shape
    if n_actions is None:
        n_actions = len(Action)

    per_run_entropy = np.full((n_runs, n_steps), np.nan, dtype=float)

    for run_idx in range(n_runs):
        run_actions = action_matrix[run_idx, :]
        for step in range(n_steps):
            start = max(0, step - window + 1)
            window_actions = run_actions[start:step + 1]
            window_actions = window_actions[np.isfinite(window_actions)]
            window_actions = window_actions[window_actions >= 0]
            if window_actions.size == 0:
                per_run_entropy[run_idx, step] = 0.0
                continue

            counts = np.bincount(window_actions.astype(int), minlength=n_actions)
            total = counts.sum()
            if total == 0:
                per_run_entropy[run_idx, step] = 0.0
                continue

            proportions = counts / total
            proportions = proportions[proportions > 0]
            per_run_entropy[run_idx, step] = -np.sum(proportions * np.log(proportions))

    mean_entropy = np.nanmean(per_run_entropy, axis=0)
    run_std = np.nanstd(per_run_entropy, axis=0, ddof=1)

    mean_entropy = np.nan_to_num(mean_entropy, nan=0.0, posinf=0.0, neginf=0.0)
    run_std = np.nan_to_num(run_std, nan=0.0, posinf=0.0, neginf=0.0)

    return mean_entropy, run_std, per_run_entropy


def compute_condition_entropy(step_values, condition_data):
    """Compute entropy curves for every condition."""
    entropy_curves = {}
    for condition, action_matrix in condition_data.items():
        mean_entropy, run_std, per_run_entropy = compute_behavioral_entropy(action_matrix)
        entropy_curves[condition] = {
            "mean": mean_entropy,
            "std": run_std,
            "per_run": per_run_entropy,
        }
    return entropy_curves


def plot_entropy_comparison(step_values, db_entropy, nb_entropy, output_path):
    """Create the DB and NB entropy comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    #Not adding actions of learned helplessness and apathy because they are not possible to do with the exploration
    n_actions = len(Action)
    max_entropy = np.log(n_actions)

    datasets = [
        (axes[0], db_entropy, "DB Agents"),
        (axes[1], nb_entropy, "NB Agents"),
    ]

    for axis, entropy_data, title in datasets:
        for condition in CONDITIONS:
            condition_curves = entropy_data.get(condition)
            if condition_curves is None or condition_curves["mean"] is None:
                continue

            entropy_curve = condition_curves["mean"]
            entropy_std = condition_curves["std"]

            axis.plot(
                step_values,
                entropy_curve,
                linewidth=2,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
            axis.fill_between(
                step_values,
                entropy_curve - entropy_std,
                entropy_curve + entropy_std,
                color=CONDITION_COLORS[condition],
                alpha=0.2,
            )

        axis.axhline(
            y=max_entropy,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Random baseline (max entropy = {max_entropy:.2f})",
        )
        axis.axhline(
            y=0,
            color="grey",
            linestyle=":",
            linewidth=1.5,
            label="Deterministic (entropy = 0)",
        )

        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_xlabel("Training Step", fontsize=12)
        axis.set_ylabel("Behavioral Entropy", fontsize=12)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Figure saved to: {output_path}")
    plt.close()


def main():
    print("=" * 72)
    print("Figure 2v2 Behavioral Entropy")
    print("=" * 72)

    print("\nSearching for series CSV files...")
    db_files, nb_files = find_series_csv_files(WORKSPACE_ROOT)
    print(f"Found {len(db_files)} DB CSV files and {len(nb_files)} NB CSV files")

    if not db_files and not nb_files:
        print("ERROR: No series CSV files found.")
        return

    print("\nProcessing DB files...")
    db_steps, db_condition_data = load_condition_action_data(db_files)

    print("\nProcessing NB files...")
    nb_steps, nb_condition_data = load_condition_action_data(nb_files)

    if db_steps is None or nb_steps is None:
        print("ERROR: Unable to load step values from the CSV files.")
        return

    db_entropy = compute_condition_entropy(db_steps, db_condition_data)
    nb_entropy = compute_condition_entropy(nb_steps, nb_condition_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"figure2v2BehavioralEntropy_{timestamp}.png")

    print("\nGenerating entropy plots...")
    plot_entropy_comparison(db_steps, db_entropy, nb_entropy, output_file)

    print("\nDone.")


if __name__ == "__main__":
    main()
