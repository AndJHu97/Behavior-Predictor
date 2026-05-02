"""
figure2v2.py - Aggregate multiple series CSV files by agent type and visualize convergence.

This script reads all DB and NB series CSV files, aggregates them by agent type,
computes rolling means and SEM confidence ribbons, and produces convergence plots
for loss and reward across training steps.

Usage:
    python figure2v2.py
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
WINDOW_SIZE = 50  # Rolling window size for smoothing
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(WORKSPACE_ROOT, "Figures")
OUTPUT_DIR = FIGURES_DIR

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_series_csv_files(workspace_root):
    """Find all series CSV files grouped by agent type (DB and NB)."""
    db_files = glob.glob(os.path.join(workspace_root, "multiple_runs_series_DB_*.csv"))
    nb_files = glob.glob(os.path.join(workspace_root, "multiple_runs_series_NB_*.csv"))
    
    # Filter Archive_Batch_Runs out if desired (keeping main folder files only)
    db_files = [f for f in db_files if "Archive" not in f]
    nb_files = [f for f in nb_files if "Archive" not in f]
    
    return sorted(db_files), sorted(nb_files)


def load_and_process_csv(csv_file, agent_type):
    """
    Load CSV and extract per-run per-step loss/reward data.
    
    Args:
        csv_file: Path to CSV file
        agent_type: "DB" or "NB"
    
    Returns:
        Dictionary with loss and reward arrays (shape: [n_steps, n_runs])
    """
    df = pd.read_csv(csv_file)
    
    # Find all run columns (e.g., run1_l_loss, run1_db_loss, etc.)
    run_nums = set()
    for col in df.columns:
        if col.startswith("run") and "_" in col:
            run_num = int(col.split("_")[0][3:])  # Extract number from "runN"
            run_nums.add(run_num)
    
    run_nums = sorted(run_nums)
    n_steps = len(df)
    n_runs = len(run_nums)
    
    # Initialize arrays for loss and reward
    loss_array = np.zeros((n_steps, n_runs))
    reward_array = np.zeros((n_steps, n_runs))
    
    # For each run, combine appropriate loss/reward columns based on agent type
    for i, run_num in enumerate(run_nums):
        l_loss_col = f"run{run_num}_l_loss"
        l_reward_col = f"run{run_num}_l_reward"
        
        if agent_type == "DB":
            # DB: use l_loss + db_loss and l_reward + db_reward (ignore NB)
            db_loss_col = f"run{run_num}_db_loss"
            db_reward_col = f"run{run_num}_db_reward"
            loss_array[:, i] = df[l_loss_col].values + df[db_loss_col].values
            reward_array[:, i] = df[l_reward_col].values + df[db_reward_col].values
        else:  # NB
            # NB: use l_loss + nb_loss and l_reward + nb_reward (ignore DB)
            nb_loss_col = f"run{run_num}_nb_loss"
            nb_reward_col = f"run{run_num}_nb_reward"
            loss_array[:, i] = df[l_loss_col].values + df[nb_loss_col].values
            reward_array[:, i] = df[l_reward_col].values + df[nb_reward_col].values
    
    return {
        "loss": loss_array,
        "reward": reward_array,
        "steps": df["step"].values
    }


def compute_rolling_mean_and_sem(data_array, window=50):
    """
    Compute rolling mean and SEM (standard error of mean) across runs.
    
    Args:
        data_array: Array of shape [n_steps, n_runs]
        window: Rolling window size
    
    Returns:
        Dictionary with rolling_mean and sem arrays
    """
    n_steps, n_runs = data_array.shape

    # Compute a trailing rolling mean independently for each run, then
    # aggregate those run-level rolling means across the agent type.
    per_run_rolling_means = np.full((n_steps, n_runs), np.nan, dtype=float)

    for run_idx in range(n_runs):
        run_values = data_array[:, run_idx]

        for step in range(n_steps):
            start = max(0, step - window + 1)
            window_values = run_values[start:step + 1]
            finite_values = window_values[np.isfinite(window_values)]

            if finite_values.size > 0:
                per_run_rolling_means[step, run_idx] = float(np.mean(finite_values))

    rolling_mean = np.nanmean(per_run_rolling_means, axis=1)
    sem = np.nanstd(per_run_rolling_means, axis=1, ddof=1) / np.sqrt(np.sum(np.isfinite(per_run_rolling_means), axis=1))

    rolling_mean = np.nan_to_num(rolling_mean, nan=0.0, posinf=0.0, neginf=0.0)
    sem = np.nan_to_num(sem, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {
        "rolling_mean": rolling_mean,
        "sem": sem
    }


def aggregate_by_agent_type(csv_files, agent_type):
    """
    Aggregate all CSV files for a given agent type.
    
    Returns:
        Dictionary with aggregated loss and reward data
    """
    all_loss_data = []
    all_reward_data = []
    
    for csv_file in csv_files:
        print(f"  Processing {os.path.basename(csv_file)}...")
        data = load_and_process_csv(csv_file, agent_type)
        all_loss_data.append(data["loss"])
        all_reward_data.append(data["reward"])
    
    # Concatenate all runs from all CSVs
    if all_loss_data:
        loss_combined = np.concatenate(all_loss_data, axis=1)  # [n_steps, total_runs]
        reward_combined = np.concatenate(all_reward_data, axis=1)
        steps = data["steps"]
    else:
        loss_combined = np.array([])
        reward_combined = np.array([])
        steps = np.array([])
    
    return {
        "loss": loss_combined,
        "reward": reward_combined,
        "steps": steps
    }


def plot_convergence(db_data, nb_data, output_path):
    """
    Create convergence plots for loss and reward with SEM ribbons.
    
    Args:
        db_data: Dictionary with DB loss/reward arrays and rolling stats
        nb_data: Dictionary with NB loss/reward arrays and rolling stats
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss Convergence
    ax = axes[0]
    
    # DB loss
    ax.plot(db_data["steps"], db_data["loss_rolling_mean"], 
            label="DB Agents", linewidth=2, color="blue")
    ax.fill_between(
        db_data["steps"],
        db_data["loss_rolling_mean"] - db_data["loss_sem"],
        db_data["loss_rolling_mean"] + db_data["loss_sem"],
        alpha=0.3, color="blue"
    )
    
    # NB loss
    ax.plot(nb_data["steps"], nb_data["loss_rolling_mean"], 
            label="NB Agents", linewidth=2, color="orange")
    ax.fill_between(
        nb_data["steps"],
        nb_data["loss_rolling_mean"] - nb_data["loss_sem"],
        nb_data["loss_rolling_mean"] + nb_data["loss_sem"],
        alpha=0.3, color="orange"
    )
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Combined Loss (L + DB/NB)", fontsize=12)
    ax.set_title("Loss Convergence Across Training Steps", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Reward Convergence
    ax = axes[1]
    
    # DB reward
    ax.plot(db_data["steps"], db_data["reward_rolling_mean"], 
            label="DB Agents", linewidth=2, color="blue")
    ax.fill_between(
        db_data["steps"],
        db_data["reward_rolling_mean"] - db_data["reward_sem"],
        db_data["reward_rolling_mean"] + db_data["reward_sem"],
        alpha=0.3, color="blue"
    )
    
    # NB reward
    ax.plot(nb_data["steps"], nb_data["reward_rolling_mean"], 
            label="NB Agents", linewidth=2, color="orange")
    ax.fill_between(
        nb_data["steps"],
        nb_data["reward_rolling_mean"] - nb_data["reward_sem"],
        nb_data["reward_rolling_mean"] + nb_data["reward_sem"],
        alpha=0.3, color="orange"
    )
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Combined Reward (L + DB/NB)", fontsize=12)
    ax.set_title("Reward Convergence Across Training Steps", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Figure saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("Figure 2v2 - Agent Type Convergence Visualization")
    print("="*70)
    
    # Find CSV files
    print("\nSearching for series CSV files...")
    db_files, nb_files = find_series_csv_files(WORKSPACE_ROOT)
    
    if not db_files and not nb_files:
        print("ERROR: No series CSV files found!")
        return
    
    print(f"Found {len(db_files)} DB CSV files and {len(nb_files)} NB CSV files")
    
    # Process DB files
    print("\n" + "-"*70)
    print("Processing DB Agent Data...")
    print("-"*70)
    if db_files:
        db_combined = aggregate_by_agent_type(db_files, "DB")
        db_loss_stats = compute_rolling_mean_and_sem(db_combined["loss"], window=WINDOW_SIZE)
        db_reward_stats = compute_rolling_mean_and_sem(db_combined["reward"], window=WINDOW_SIZE)
        db_data = {
            "steps": db_combined["steps"],
            "loss_rolling_mean": db_loss_stats["rolling_mean"],
            "loss_sem": db_loss_stats["sem"],
            "reward_rolling_mean": db_reward_stats["rolling_mean"],
            "reward_sem": db_reward_stats["sem"]
        }
        print(f"✓ DB data processed: {db_combined['loss'].shape[1]} runs across {len(db_files)} files")
        print(f"  Loss range: [{db_loss_stats['rolling_mean'].min():.4f}, {db_loss_stats['rolling_mean'].max():.4f}]")
        print(f"  Reward range: [{db_reward_stats['rolling_mean'].min():.4f}, {db_reward_stats['rolling_mean'].max():.4f}]")
    
    # Process NB files
    print("\n" + "-"*70)
    print("Processing NB Agent Data...")
    print("-"*70)
    if nb_files:
        nb_combined = aggregate_by_agent_type(nb_files, "NB")
        nb_loss_stats = compute_rolling_mean_and_sem(nb_combined["loss"], window=WINDOW_SIZE)
        nb_reward_stats = compute_rolling_mean_and_sem(nb_combined["reward"], window=WINDOW_SIZE)
        nb_data = {
            "steps": nb_combined["steps"],
            "loss_rolling_mean": nb_loss_stats["rolling_mean"],
            "loss_sem": nb_loss_stats["sem"],
            "reward_rolling_mean": nb_reward_stats["rolling_mean"],
            "reward_sem": nb_reward_stats["sem"]
        }
        print(f"✓ NB data processed: {nb_combined['loss'].shape[1]} runs across {len(nb_files)} files")
        print(f"  Loss range: [{nb_loss_stats['rolling_mean'].min():.4f}, {nb_loss_stats['rolling_mean'].max():.4f}]")
        print(f"  Reward range: [{nb_reward_stats['rolling_mean'].min():.4f}, {nb_reward_stats['rolling_mean'].max():.4f}]")
    
    # Generate plots
    if db_files and nb_files:
        print("\n" + "-"*70)
        print("Generating Convergence Plots...")
        print("-"*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"figure2v2_convergence_{timestamp}.png")
        
        plot_convergence(db_data, nb_data, output_file)
    else:
        print("\nWARNING: Missing DB or NB data. Skipping plot generation.")
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
