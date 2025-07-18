import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.show()

def plot_curves(
    relL_values, relDB_values, relNB_values,
    sitL_values, sitDB_values, sitNB_values,
    action_values, survival_rounds_values,
    lLoss_values, dbLoss_values, nbLoss_values,
    sit_type, threat_action_values, ally_action_values, prey_action_values
):
    fig = plt.figure(figsize=(18, 30), constrained_layout=True)

    plt.subplot(6, 3, 1)
    plt.plot(relL_values, color = 'brown')
    plt.title('Character.relL')

    plt.subplot(6, 3, 2)
    plt.plot(relDB_values, color = 'navy')
    plt.title('Character.relDB')

    plt.subplot(6, 3, 3)
    plt.plot(relNB_values, color = 'lightgreen')
    plt.title('Character.relNB')

    plt.subplot(6, 3, 4)
    plt.plot(sitL_values, color = 'brown')
    plt.title('Situation sitL')

    plt.subplot(6, 3, 5)
    plt.plot(sitDB_values, color = "navy")
    plt.title('Situation sitDB')

    plt.subplot(6, 3, 6)
    plt.plot(sitNB_values, color = 'lightgreen')
    plt.title('Situation sitNB')

    plt.subplot(6, 3, 7)
    plt.scatter(range(len(action_values)), action_values)
    plt.title('Action')

    plt.subplot(6, 3, 8)
    plt.plot(survival_rounds_values)
    plt.title('Rounds Survived')

    plt.subplot(6, 3, 9)
    plt.scatter(range(len(lLoss_values)), lLoss_values, color= 'brown')
    plt.title('Livelihood Loss')

    plt.subplot(6, 3, 10)
    plt.scatter(range(len(dbLoss_values)), dbLoss_values, color='navy')
    plt.title('Defensive Belonging Loss')

    plt.subplot(6, 3, 11)
    plt.scatter(range(len(nbLoss_values)), nbLoss_values, color = 'lightgreen')
    plt.title('Nurturing Belonging Loss')

    plt.subplot(6, 3, 12)
    plt.scatter(range(len(sit_type)), sit_type)
    plt.title('Situation Type')
    plt.yticks([0, 1, 2], ['Threat','Ally','Prey'])


    plt.subplot(6, 3, 13)
    plt.scatter(range(len(threat_action_values)), threat_action_values, color='red')
    plt.title('Threat Actions')
    plt.yticks([-2, -1, 0, 1, 2, 3, 4], ['Helpless','Nothing Worth Doing','Fight', 'Flee', 'Befriend', 'Chase', 'Cry'])

    plt.subplot(6, 3, 14)
    plt.scatter(range(len(ally_action_values)), ally_action_values, color='blue')
    plt.title('Ally Actions')
    plt.yticks([-2, -1, 0, 1, 2, 3, 4], ['Helpless','Nothing Worth Doing','Fight', 'Flee', 'Befriend', 'Chase', 'Cry'])

    plt.subplot(6, 3, 15)
    plt.scatter(range(len(prey_action_values)), prey_action_values, color='green')
    plt.title('Prey Actions')
    plt.yticks([-2, -1, 0, 1, 2, 3, 4], ['Helpless','Nothing Worth Doing','Fight', 'Flee', 'Befriend', 'Chase', 'Cry'])



    plt.show(block=False)

def plot_complex_psychology_curves(complex_psychology_values, title):
    if not complex_psychology_values:
        print(f"⚠️ Skipping plot for '{title}': No data to plot.")
        return

    sns.set(style="whitegrid")
    
    # Create DataFrame
    df = pd.DataFrame(complex_psychology_values)
    
    # Melt character reward values
    df_char = df.melt(
        id_vars=["sit_type"],
        value_vars=["relL", "relDB", "relNB"],
        var_name="RewardType",
        value_name="Value"
    )
    df_char["Source"] = "Character"
    
    # Melt situation reward values
    df_sit = df.melt(
        id_vars=["sit_type"],
        value_vars=["sitL", "sitDB", "sitNB"],
        var_name="RewardType",
        value_name="Value"
    )
    df_sit["Source"] = "Situation"
    
    # Combine
    df_combined = pd.concat([df_char, df_sit])
    df_combined["Combined"] = df_combined["Source"] + "_" + df_combined["RewardType"]
    
    # Group for occurrence count
    occurrence_counts = df_combined.groupby(["sit_type", "RewardType", "Source"]).size().reset_index(name="Count")
    
    # Create figure
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    # 🆕 Add total occurrences at the top of the entire figure
    total_occurrences = len(complex_psychology_values)
    fig.suptitle(
    f"{title} — Total Occurrences: {total_occurrences}",
    fontsize=16,
    fontweight='bold',
    y=0.99  # ⬅️ move it higher above the subplots
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # ⬅️ leave space at the top
    
    # Stripplot: Individual values
    sns.stripplot(
        data=df_combined,
        x="sit_type",
        y="Value",
        hue="Combined",
        dodge=True,
        jitter=True,
        palette="muted",
        ax=ax[0]
    )

    # Overlay estimated and actual reward (mean values per sit_type)
    est_means = df.groupby("sit_type")["estimated_reward"].mean()
    act_means = df.groupby("sit_type")["actual_reward"].mean()
    
    # Plot estimated/actual reward as points
    for i, sit in enumerate(est_means.index):
        ax[0].scatter(i, est_means[sit], color='black', marker='X', s=100, label='Estimated Reward' if i == 0 else "")
        ax[0].scatter(i, act_means[sit], color='red', marker='D', s=100, label='Actual Reward' if i == 0 else "")

    ax[0].set_title(title + " (Individual Values)")
    ax[0].set_ylabel("Value")
    ax[0].set_xlabel("Situation Type")
    ax[0].legend(title="Source_RewardType / Reward", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Barplot: Occurrence counts
    sns.barplot(
        data=occurrence_counts,
        x="sit_type",
        y="Count",
        hue="Source",
        palette="deep",
        ax=ax[1]
    )
    
    ax[1].set_title("Occurrences of " + title + " Types per Situation")
    ax[1].set_ylabel("Number of Occurrences")
    ax[1].set_xlabel("Situation Type")
    ax[1].legend(title="Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show(block=False) #allows all windows to be loaded at once


def display_occurrence_counts_plot(**kwargs):
    """
    Displays a summary in a popup window using matplotlib.
    Keys starting with 'H:' or 'H_' are treated as bold headers but still show counts.
    """
    fig, ax = plt.subplots(figsize=(6, len(kwargs) * 0.6 + 1))  # Slightly taller figure
    ax.axis('off')

    y_start = 1.0
    line_height = 0.1  # Increased from 0.08 to 0.1 for more spacing

    for i, (name, count) in enumerate(kwargs.items()):
        is_header = str(name).startswith("H:") or str(name).startswith("H_")
        clean_name = name[2:].replace("_", " ") if is_header else name.replace("_", " ")

        fontsize = 14 if is_header else 11
        weight = 'bold' if is_header else 'normal'
        text = f"{clean_name}: {count}"

        ax.text(
            0.5,
            y_start - i * line_height,
            text,
            ha='center',
            va='top',
            fontsize=fontsize,
            fontweight=weight,
            family='monospace'
        )

    plt.title("📊 Occurrence Summary", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show(block=False)



