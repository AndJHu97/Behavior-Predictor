# Behavior Predictor

## Overview
This project simulates how an agent chooses behavior across randomized situations:
- Threat
- Ally
- Prey

The model is a contextual, single-step decision system. It does not optimize long-horizon trajectories. Instead, each encounter is treated as a local decision based on current state values.

Core tracked dimensions:
- Livelihood (L): physical well-being and survivability
- Defensive Belonging (DB): social value from protection, power, and deterrence
- Nurturing Belonging (NB): social value from support, care, and cooperation

The simulation trains action-specific value networks and records both:
- Basic behavior frequencies (fight, flee, befriend, chase, cry)
- Higher-level behavioral patterns (for example, protective behavior, dangerous trust, learned helplessness)

## How The Code Works

### High-level flow
1. UI collects parameters in simulation.py (environment distributions, risk/reward preferences, training settings).
2. main(...) runs a training loop over Training_Episodes.
3. At each episode:
  - Build state from character and current situation
  - Select action (exploration during learning period, then exploitation)
  - Apply environment response via situation-specific reward logic
  - Train selected value networks on observed reward
  - Log metrics, losses, and complex behavior counters
4. At the end of a run, plots and advice exports are generated (when plotting is enabled).
5. Multiple-run mode aggregates statistics across repeated runs and exports a CSV summary.

### Learning approach
- The agent keeps separate value networks per action and per reward type (L, DB, NB).
- State includes normalized L/DB/NB and situation stats plus one-hot situation type.
- Action selection combines:
  - Exploration during the learning period
  - Exploitation by choosing the best predicted reward among L and the selected belonging type (mainB)
- Risk and reward filters can suppress actions if predicted downside exceeds thresholds.

### Complex behavior metrics
Simulation logic also tracks composite psychological patterns such as:
- boredom_maladaptive
- learned_helplessness
- apathy
- positive_mindset
- community_trusting_vulnerability
- fearful_withdrawn_relationship
- willingness_to_flee
- self_destructive_anger
- bully_behavior
- protective_behavior
- healthy_friendliness
- dangerous_trust
- over_friendliness
- hopefulness
- cynical

## Project Structure (Relevant Files)

### Main entry point
- simulation.py: Tkinter app, training loop, multiple-run evaluation, plotting calls, save/load model toggles, advice export integration.

### Core modeling code
- agent.py: Character and Agent classes, action selection policy, state encoding, memory handling, short-memory learning, model save/load, training stat save.
- situations.py: Environment definitions for Threat, Ally, Prey and reward/update rules for each action.
- ValueNetwork.py: Feed-forward value network used per action and reward channel.

### Utilities and supporting modules
- helper.py: Plotting and summary visualization functions.
- quiz_mode.py: Personality test UI that fills simulation parameters from questionnaire answers.
- advice/advice_generator.py: Loads advice rules, evaluates score-based conditions, exports filtered advice (CSV + TXT).

### Alternate/legacy model experiments
- CBmodel.py: Contextual bandit prototype (marked as not used).
- PGmodel.py: Policy gradient network/trainer prototype.
- model/policy_model.pth: Policy model artifact from PG experiments.

### Data and generated artifacts
- advice/advice.csv: Advice rule table used by advice generator.
- saved_models/: Saved value-network weights by model name.
- saved_stats/: Saved training parameter snapshots by model name.
- multiple_runs_*.csv: Batch run exports from multiple simulation runs.
- filtered_advice.csv and advice/*.csv/.txt: Advice outputs.

### Notebook
- jupytermodel.ipynb: Early sandbox notebook for model experimentation (not the primary runtime path).

### Analysis scripts
- FinalProject/: Post-simulation analysis scripts and exported figure/stat artifacts.
- FinalProject/figure1_environment_behavior.py: Generates environment behavior comparisons and summary outputs.
- FinalProject/figure2_role_differences.py: Role-based differences across model behaviors.
- FinalProject/figure3_baseline_adjusted.py: Baseline-adjusted analysis workflow.
- FinalProject/figure4_complex_behaviors.py and FinalProject/figure4b_adaptive_behaviors.py: Complex behavior and adaptive behavior analysis.
- FinalProject/figure5_*.py, FinalProject/figure6_behavioral_collapse.py, FinalProject/figure7*.py: Emergence probability, collapse, and protective/maladaptive dynamics studies.

## Requirements
Python 3.9+ is recommended.

External packages used by the code:
- numpy
- pandas
- matplotlib
- seaborn
- torch

Tkinter is used for the UI and is included with standard Python on most installations.

## Setup

### Windows PowerShell
1. Create virtual environment:
  - py -m venv venv
2. Activate virtual environment:
  - .\venv\Scripts\Activate.ps1
3. Install dependencies:
  - pip install numpy pandas matplotlib seaborn torch

Optional for notebook work:
- pip install notebook jupyterlab

## Running The Simulation
From the repository root:

1. Activate venv:
  - .\venv\Scripts\Activate.ps1
2. Launch app:
  - python simulation.py

Then in the UI you can:
- Run a single simulation with Start Simulation
- Run batch trials with Run Multiple Simulations
- Toggle Predict Action Mode (disables learning updates)
- Save Model / Load Model
- Open Personality Test to prefill settings

## Running Analysis Scripts (FinalProject)
Most figure scripts can be run directly after dependencies are installed:

1. Activate venv:
  - .\venv\Scripts\Activate.ps1
2. Run a script:
  - python FinalProject/figure1_environment_behavior.py
  - python FinalProject/figure2_role_differences.py

Each script may expect specific CSV inputs already present in FinalProject/.

## Typical Outputs
- On-screen plots for trajectory, actions, and loss trends.
- Console loss summary with mean plus/minus 95% CI for L/DB/NB loss.
- CSV export for multiple-run statistics (timestamped file).
- Advice export files in advice/ based on behavioral score aggregates.

## Notes
- This project is intentionally designed around immediate, local decision quality rather than long-term planning.
- The included CBmodel.py and PGmodel.py are useful references but are not the default simulation path.

