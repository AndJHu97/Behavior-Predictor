import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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



    plt.show()

def plot_complex_psychology_curves(boredom_maladaptive_values):
    sns.set(style="whitegrid")
    
    # Create DataFrame
    df_boredom_maladaptive_values = pd.DataFrame(boredom_maladaptive_values)
    
    # Melt character reward values
    df_melted_character = df_boredom_maladaptive_values.melt(
        id_vars="sit_type",
        value_vars=["relL", "relDB", "relNB"],
        var_name="RewardType",
        value_name="Value"
    )
    df_melted_character["Source"] = "Character"
    
    # Melt situation reward values
    df_melted_sit = df_boredom_maladaptive_values.melt(
        id_vars="sit_type",
        value_vars=["sitL", "sitDB", "sitNB"],
        var_name="RewardType",
        value_name="Value"
    )
    df_melted_sit["Source"] = "Situation"
    
    # Combine
    df_combined = pd.concat([df_melted_character, df_melted_sit])
    
    # Combine Source and RewardType for better hue labeling (optional)
    df_combined["Combined"] = df_combined["Source"] + "_" + df_combined["RewardType"]

    # Count the number of times each (sit_type, RewardType, Source) occurs
    occurrence_counts = df_combined.groupby(["sit_type", "RewardType", "Source"]).size().reset_index(name="Count")
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))  # 2 rows, 1 column
    
    # Plot the stripplot (scatter plot of individual values)
    sns.stripplot(
        data=df_combined,
        x="sit_type",
        y="Value",
        hue="Combined",
        dodge=True,  # separates the points by hue within each x category
        jitter=True, # spreads out points so they don't overlap
        palette="muted",
        ax=ax[0]  # Assigning to the first subplot
    )
    ax[0].set_title("Maladaptive Behaviors Out of Boredom (Individual Values)")
    ax[0].set_ylabel("Value")
    ax[0].set_xlabel("Situation Type")
    ax[0].legend(title="Source_RewardType", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot the barplot (occurrences count)
    sns.barplot(
        data=occurrence_counts,
        x="sit_type",
        y="Count",
        hue="Source",  # or hue="RewardType" if that's your priority
        palette="deep",
        ax=ax[1]  # Assigning to the second subplot
    )
    ax[1].set_title("Occurrences of Maladaptive Reward Types per Situation")
    ax[1].set_ylabel("Number of Occurrences")
    ax[1].set_xlabel("Situation Type")
    ax[1].legend(title="Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
