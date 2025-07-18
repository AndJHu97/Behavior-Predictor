import tkinter as tk
import json
import os
import numpy as np
from tkinter import messagebox
import random
from situations import Action, SituationType, Threat, Ally, Prey
from agent import Character, Agent
from helper import plot_curves, plot_complex_psychology_curves, display_occurrence_counts_plot
from advice.advice_generator import load_advice_file, filter_advice, export_advice



agent = None
#Don't need save model entry because save should be at any time but load model only matters with starting simulation
load_model_entry = False
predict_action_entry = False

# Assuming your existing imports and class definitions like Character, Threat, Ally, etc. are already included

def start_simulation():
    # Get the values from the Tkinter fields
    prob_threat = float(prob_threat_entry.get())
    prob_ally = float(prob_ally_entry.get())
    prob_prey = float(prob_prey_entry.get())
    tLowerSitL = int(tLowerSitL_entry.get())
    tHigherSitL = int(tHigherSitL_entry.get())
    tLowerSitDB = int(tLowerSitDB_entry.get())
    tHigherSitDB = int(tHigherSitDB_entry.get())
    tLowerSitNB = int(tLowerSitNB_entry.get())
    tHigherSitNB = int(tHigherSitNB_entry.get())
    
    aLowerSitL = int(aLowerSitL_entry.get())
    aHigherSitL = int(aHigherSitL_entry.get())
    aLowerSitDB = int(aLowerSitDB_entry.get())
    aHigherSitDB = int(aHigherSitDB_entry.get())
    aLowerSitNB = int(aLowerSitNB_entry.get())
    aHigherSitNB = int(aHigherSitNB_entry.get())

    pLowerSitL = int(pLowerSitL_entry.get())
    pHigherSitL = int(pHigherSitL_entry.get())
    pLowerSitDB = int(pLowerSitDB_entry.get())
    pHigherSitDB = int(pHigherSitDB_entry.get())
    pLowerSitNB = int(pLowerSitNB_entry.get())
    pHigherSitNB = int(pHigherSitNB_entry.get())

    #society
    societyL = int(societyL_entry.get())
    societyNB = int(societyNB_entry.get())
    societyDB = int(societyDB_entry.get())

    Risk_Aversion = float(risk_aversion_entry.get())
    Risk_Threshold = float(risk_threshold_entry.get())
    Reward_Inclination = float(reward_inclination_entry.get())
    Reward_Threshold = float(reward_threshold_entry.get())
    MainB = mainB_var.get()  # Get the value from the dropdown menu
    Training_Episodes = int(training_episodes_entry.get())
    Learning_Period = int(learning_period_entry.get())
    Lr = float(lr_entry.get())
    model_name = model_name_entry.get()

    if predict_action_entry:
        Lr = 0
        Learning_Period = 0
    
    # Call the main function with these values
    main(prob_threat, prob_ally, prob_prey, tLowerSitL, tHigherSitL, tLowerSitDB, tHigherSitDB, tLowerSitNB, tHigherSitNB,
         aLowerSitL, aHigherSitL, aLowerSitDB, aHigherSitDB, aLowerSitNB, aHigherSitNB,
         pLowerSitL, pHigherSitL, pLowerSitDB, pHigherSitDB, pLowerSitNB, pHigherSitNB,
         societyL, societyNB, societyDB,
         Risk_Aversion, Risk_Threshold, Reward_Inclination, Reward_Threshold, MainB, Training_Episodes, Lr, Learning_Period, model_name)

def main(prob_threat, prob_ally, prob_prey, tLowerSitL, tHigherSitL, tLowerSitDB, tHigherSitDB, tLowerSitNB, tHigherSitNB, 
         aLowerSitL, aHigherSitL, aLowerSitDB, aHigherSitDB, aLowerSitNB, aHigherSitNB, 
         pLowerSitL, pHigherSitL, pLowerSitDB, pHigherSitDB, pLowerSitNB, pHigherSitNB,
         societyL, societyNB, societyDB,
         Risk_Aversion, Risk_Threshold, Reward_Inclination, Reward_Threshold, MainB, Training_Episodes, LR, Learning_Period, model_name):
    # Define the character and the initial situation
    absL = 100
    absNB = 100
    absDB = 100
    tSitL = random.randint(tLowerSitL, tHigherSitL)
    tSitDB = random.randint(tLowerSitDB, tHigherSitDB)
    tSitNB = random.randint(tLowerSitNB, tHigherSitNB)
    aSitL = random.randint(aLowerSitL, aHigherSitL)
    aSitDB = random.randint(aLowerSitDB, aHigherSitDB)
    aSitNB = random.randint(aLowerSitNB, aHigherSitNB)
    pSitL = random.randint(pLowerSitL, pHigherSitL)
    pSitDB = random.randint(pLowerSitDB, pHigherSitDB)
    pSitNB = random.randint(pLowerSitNB, pHigherSitNB)

    character = Character(risk_aversion= Risk_Aversion, risk_threshold=Risk_Threshold, reward_inclination=Reward_Inclination, reward_threshold= Reward_Threshold, absL=absL, absNB=absNB, absDB = absDB, mainB = MainB)
    randomChance = random.random()
    if randomChance < prob_threat:
        situation = Threat(sitL=tSitL, sitDB=tSitDB, sitNB = tSitNB, sitType=SituationType.Threat, societyL=societyL, societyDB=societyDB, societyNB=societyNB)
    elif randomChance < prob_threat + prob_ally:
        situation = Ally(sitL = aSitL, sitDB= aSitDB, sitNB = aSitNB, sitType=SituationType.Ally)
    else:
        situation = Prey(sitL = pSitL, sitDB= pSitDB, sitNB = pSitNB, sitType=SituationType.Prey)
    global agent
    agent = Agent(actions=list(Action), Lr=LR, Learning_Period= Learning_Period)
    
    if(load_model_entry):
        agent.load_models(model_name)

    relL_values = []
    relNB_values = []
    relDB_values = []
    sitL_values = []
    sitNB_values = []
    sitDB_values = []
    sit_types = []
    action_values = []
    survival_rounds_values = []
    rewards_values = []
    nbLoss_values = []
    dbLoss_values = []
    lLoss_values = []
    threat_action_values = []
    ally_action_values = []
    prey_action_values = []
    boredom_maladaptive_values = []
    positive_mindset_values = []
    community_trusting_vulnerability_values = []
    fearful_withdrawn_relationship_values = []
    detached_withdrawn_relationship_values = []
    aggressive_withdrawn_relationship_values = []
    willingness_to_flee_values = []
    self_destructive_anger_values = []
    bully_behavior_values = []
    protective_behavior_values = []
    healthy_friendliness_values = []
    dangerous_trust_values = []
    over_friendliness_values = []
    hopefulness_values = []
    cynical_values = []

    rounds_encountered = 0
    for episode in range(Training_Episodes):
        state = agent.get_state(character, situation)

        #if estimated_action_reward is np.nan, this means it is a random selection
        action, estimated_action_reward, type_of_action_stat = agent.select_action(character, state, rounds_encountered)

        relL_values.append(character.relL)
        relNB_values.append(character.relNB)
        relDB_values.append(character.relDB)
        sitL_values.append(situation.sitL)
        #Change to right situation 
        sitNB_values.append(situation.sitNB) 
        sitDB_values.append(situation.sitDB)
        sit_types.append(situation.sitType.value)
        action_values.append(action)

        if(situation.sitType.value == 0):
            threat_action_values.append(action)
        elif(situation.sitType.value == 1):
            ally_action_values.append(action)
        elif(situation.sitType.value == 2):
            prey_action_values.append(action)
            
        blStore = []
        lReward, dbReward, nbReward, death, survival_rounds = situation.process_action(character, action)
        survival_rounds_values.append(survival_rounds)
        rewards_values.append(dbReward)
        blStore = agent.train_short_memory(state, action, lReward, dbReward, nbReward)
        lLoss_values.append(blStore[0])
        #Loss should be from adjusted after the NN returns 3 values so it can have 3 values in blStore for L, NB, and DB
        dbLoss_values.append(blStore[1])
        nbLoss_values.append(blStore[2])
        agent.remember(state, action, lReward, dbReward, nbReward, agent.lSelectedActionModel, agent.dbSelectedActionModel, agent.nbSelectedActionModel)
        #character.set_stats(character.relL + 5, character.relDB, character.relNB)
       
        #COMPLEX PSYCHOLOGY

        #Get actual rewards
        #If actual_reward = np.nan then type_of_action is none and is due to depression or helplessness or random
        actual_reward = np.nan
        if type_of_action_stat == "L":
            actual_reward = lReward
        elif type_of_action_stat == "DB":
            actual_reward = dbReward
        elif type_of_action_stat == "NB":
            actual_reward = nbReward


        #Maladaptive Behavior because better than boredom
        #Negative rewards and not helpless nor not worth doing nothing
        if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
            if estimated_action_reward < 0 and action != -1 and action != -2:
                boredom_maladaptive_values.append({
                    'sit_type': situation.sitType.value,
                    'action': action,
                    'relNB': character.relNB,
                    'relDB': character.relDB,
                    'relL': character.relL,
                    'sitNB': situation.sitNB,
                    'sitDB': situation.sitDB,
                    'sitL': situation.sitL,
                    'estimated_reward': estimated_action_reward,
                    'actual_reward': actual_reward
                })

        #Positive/Optimistic Mindset. Prey, Chase, and estimated reward < actual reward
        if situation.sitType.value == 2 and action == 3:
            if not np.isnan(actual_reward) and not np.isnan(actual_reward):
                if actual_reward > estimated_action_reward and estimated_action_reward > 0:
                    positive_mindset_values.append({
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    })


        #Community Trusting Vulnerability
        #Able to cry when see threat with belief it will result in something good (estimated reward )
        if situation.sitType.value == 0 and action == 4:
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                if estimated_action_reward > 0:
                    community_trusting_vulnerability_values.append({
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    })

        #Fearful Relationship 
        #Check if it is an ally with the fearful actions needed (cry or flee)
        if situation.sitType.value == 1 and (action == 1 or action == 4):
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                fearful_withdrawn_relationship_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #Detached
        #Check if helpless or not worth doing (depression) with allies
        if situation.sitType.value == 1 and (action == -1 or action == -2):
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                detached_withdrawn_relationship_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #Aggressive
        #Check if fight or chase with allies
        if situation.sitType.value == 1 and (action == 0 or action == 3):
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                aggressive_withdrawn_relationship_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #Willingness to Run
        if situation.sitType.value == 0 and action == 1:
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                willingness_to_flee_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #When fighting but does not expect anything good from it
        if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
            if action == 0 and estimated_action_reward < 0:
                self_destructive_anger_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )
        #Bully
        #When fighting and feeling good and not justified for fighting off threats
        if action == 0 and estimated_action_reward > 0 and situation.sitType.value != 0:
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                bully_behavior_values.append(
                     {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #Protective
        #Fighting threats
        if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
            if action == 0 and situation.sitType.value == 0:
                protective_behavior_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #Healthy Friendliness
        #Befriending allies
        if action == 2 and situation.sitType.value == 1:
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                healthy_friendliness_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #Dangerous Trust
        #Befriending Threats
        if action == 2 and situation.sitType.value == 0:
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                dangerous_trust_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #Over friendliness
        #Befriending everything but allies
        if action == 2 and situation.sitType.value != 1:
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                over_friendliness_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        #Hopeful
        if estimated_action_reward > 0:
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                hopefulness_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )
        
        #Cynical, expect negative rewards
        if estimated_action_reward < 0:
            if not np.isnan(estimated_action_reward) and not np.isnan(actual_reward):
                cynical_values.append(
                    {
                        'sit_type': situation.sitType.value,
                        'action': action,
                        'relNB': character.relNB,
                        'relDB': character.relDB,
                        'relL': character.relL,
                        'sitNB': situation.sitNB,
                        'sitDB': situation.sitDB,
                        'sitL': situation.sitL,
                        'estimated_reward': estimated_action_reward,
                        'actual_reward': actual_reward
                    }
                )

        if death:
            print(f"Character died after {survival_rounds} rounds")
            character = Character(risk_aversion= Risk_Aversion, risk_threshold= Risk_Threshold, reward_inclination = Reward_Inclination, reward_threshold = Reward_Threshold,  absL=absL, absNB=absNB, absDB = absDB, mainB = MainB)
            #not long term because not very human like
            #blStore = agent.train_long_memory()
            #lLoss_values.append(blStore[0])
            #bLoss_values.append(blStore[1])
        tSitL = random.randint(tLowerSitL, tHigherSitL)
        tSitDB = random.randint(tLowerSitDB, tHigherSitDB)
        tSitNB = random.randint(tLowerSitNB, tHigherSitNB)
        aSitL = random.randint(aLowerSitL, aHigherSitL)
        aSitDB = random.randint(aLowerSitDB, aHigherSitDB)
        aSitNB = random.randint(aLowerSitNB, aHigherSitNB)
        #character = Character(absL=absL, absB=absB)
        randomChance = random.random()
        if randomChance < prob_threat:
            #Can incorporate the society aspect here
            situation = Threat(sitL=tSitL, sitDB=tSitDB, sitNB = tSitNB, sitType=SituationType.Threat, societyL=societyL, societyDB=societyDB, societyNB=societyNB)
        elif randomChance < prob_threat + prob_ally:
            situation = Ally(sitL = aSitL, sitDB= aSitDB, sitNB = aSitNB, sitType=SituationType.Ally)
        else:
            situation = Prey(sitL = pSitL, sitDB= pSitDB, sitNB = pSitNB, sitType=SituationType.Prey)

        rounds_encountered += 1

    

    plot_curves(relL_values, relDB_values, relNB_values, sitL_values, sitDB_values, sitNB_values, action_values, survival_rounds_values, lLoss_values, dbLoss_values, nbLoss_values, sit_types, threat_action_values, ally_action_values, prey_action_values)
    
    pr_guardian_value = len(bully_behavior_values) + len(self_destructive_anger_values) + len(protective_behavior_values)
    pr_sustainer_value = len(community_trusting_vulnerability_values) + len(willingness_to_flee_values) + len(healthy_friendliness_values)
    relationship_engaging_value = len(over_friendliness_values) + len(dangerous_trust_values) + len(healthy_friendliness_values)
    relationship_withdrawn_value = len(fearful_withdrawn_relationship_values) + len(detached_withdrawn_relationship_values) + len(aggressive_withdrawn_relationship_values)
    drive_productive_value = len(hopefulness_values) + len(positive_mindset_values)
    drive_destructive_value = len(boredom_maladaptive_values) + len(cynical_values) * 2

    display_occurrence_counts_plot(
    **{
        "H:Protective Role - Guardian": pr_guardian_value,
        "Self Destructive Anger": len(self_destructive_anger_values),
        "Bully Behavior": len(bully_behavior_values),
        "Protective Behavior": len(protective_behavior_values),
        "H:Protective Role - Sustainer": pr_sustainer_value,
        "Community Trusting Vulnerability Values": len(community_trusting_vulnerability_values),
        "Willingness To Flee": len(willingness_to_flee_values),
        "Healthy Friendliness (A)": len(healthy_friendliness_values),
        "H:Relational Mode - Engaging": relationship_engaging_value,
        "Overfriendliness": len(over_friendliness_values),
        "Dangerous Trust": len(dangerous_trust_values),
        "Healthy Friendliness (B)": len(healthy_friendliness_values),
        "H:Relational Mode - Withdrawn": relationship_withdrawn_value,
        "Fearful Relationship Behavior": len(fearful_withdrawn_relationship_values),
        "Aggressive Relationship Behavior": len(aggressive_withdrawn_relationship_values),
        "Detached Relationship Behavior": len(detached_withdrawn_relationship_values),
        "H:Drive Style - Productive": drive_productive_value,
        "Hopeful": len(hopefulness_values),
        "Positive Mindset in Goal Pursuit": len(positive_mindset_values),
        "H:Drive Style - Destructive": drive_destructive_value,
        "Drive Style - Destructive (original value)": len(boredom_maladaptive_values) + len(cynical_values),
        "Cynical": len(cynical_values),
        "Cynical (Weighted Value)": len(cynical_values) * 2,
        "Maladaptive Behaviors Out Of Boredom": len(boredom_maladaptive_values),
    }
    
    )
    
    plot_complex_psychology_curves(boredom_maladaptive_values, "Maladaptive Behaviors Out Of Boredom")
    plot_complex_psychology_curves(positive_mindset_values, "Positive Mindset In Goal Pursuit")
    plot_complex_psychology_curves(community_trusting_vulnerability_values, "Community Trusting Behavior With Vulnerability")
    plot_complex_psychology_curves(fearful_withdrawn_relationship_values, "Avoidant Personality Towards Relationships")
    plot_complex_psychology_curves(willingness_to_flee_values, "Willingness To Flee")
    plot_complex_psychology_curves(self_destructive_anger_values, "Self Destructive Anger")
    plot_complex_psychology_curves(bully_behavior_values, "Bully Behavior")
    plot_complex_psychology_curves(protective_behavior_values, "Protective Behaviors")
    plot_complex_psychology_curves(healthy_friendliness_values, "Healthy Friendliness")
    plot_complex_psychology_curves(dangerous_trust_values, "Dangerous Trust")
    plot_complex_psychology_curves(over_friendliness_values, "Over-friendliness")
    plot_complex_psychology_curves(hopefulness_values, "Hopeful World Lens")
    plot_complex_psychology_curves(cynical_values, "Cynical World Lens")

    scores = {
        "belonging_type": MainB,
        "pr_guardian": pr_guardian_value,
        "pr_sustainer": pr_sustainer_value,
        "relationship_engaging": relationship_engaging_value,
        "relationship_withdrawn": relationship_withdrawn_value,
        "drive_productive": drive_productive_value,
        "drive_destructive": drive_destructive_value
    }

    advice_df = load_advice_file("advice/advice.csv")
    filtered_advice = filter_advice(advice_df, scores)
    export_advice(filtered_advice, filename=f"advice/{model_name}.csv", text_filename=f"advice/{model_name}.txt")

# Create the main Tkinter window
root = tk.Tk()
root.title("Simulation Input")

# Create and place labels and input fields for each value with default values
tk.Label(root, text="Probability of Threat (0-1)").grid(row=0, column=0)
prob_threat_entry = tk.Entry(root)
prob_threat_entry.grid(row=0, column=1)
prob_threat_entry.insert(0, "0.33")  # Default value for Probability of Threat

tk.Label(root, text="Probability of Ally (0-1)").grid(row=1, column=0)
prob_ally_entry = tk.Entry(root)
prob_ally_entry.grid(row=1, column=1)
prob_ally_entry.insert(0, "0.33")  # Default value for Probability of Ally

tk.Label(root, text="Probability of Prey (0-1)").grid(row=2, column=0)
prob_prey_entry = tk.Entry(root)
prob_prey_entry.grid(row=2, column=1)
prob_prey_entry.insert(0, "0.33")  # Default value for Probability of Prey

tk.Label(root, text="Threat's Livelihood (Lower Bound)").grid(row=3, column=0)
tLowerSitL_entry = tk.Entry(root)
tLowerSitL_entry.grid(row=3, column=1)
tLowerSitL_entry.insert(0, "80")  # Default value for Threat's Livelihood Lower Bound

tk.Label(root, text="Threat Livelihood (Upper Bound)").grid(row=4, column=0)
tHigherSitL_entry = tk.Entry(root)
tHigherSitL_entry.grid(row=4, column=1)
tHigherSitL_entry.insert(0, "100")  # Default value for Threat Livelihood Upper Bound

tk.Label(root, text="Threat Defensive Belonging (Lower Bound)").grid(row=5, column=0)
tLowerSitDB_entry = tk.Entry(root)
tLowerSitDB_entry.grid(row=5, column=1)
tLowerSitDB_entry.insert(0, "80")  # Default value for Threat Defensive Belonging Lower Bound

tk.Label(root, text="Threat Defensive Belonging (Upper Bound)").grid(row=6, column=0)
tHigherSitDB_entry = tk.Entry(root)
tHigherSitDB_entry.grid(row=6, column=1)
tHigherSitDB_entry.insert(0, "100")  # Default value for Threat Defensive Belonging Upper Bound

tk.Label(root, text="Threat Nurturing Belonging (Lower Bound)").grid(row=7, column=0)
tLowerSitNB_entry = tk.Entry(root)
tLowerSitNB_entry.grid(row=7, column=1)
tLowerSitNB_entry.insert(0, "80")  # Default value for Threat Nurturing Belonging Lower Bound

tk.Label(root, text="Threat Nurturing Belonging (Upper Bound)").grid(row=8, column=0)
tHigherSitNB_entry = tk.Entry(root)
tHigherSitNB_entry.grid(row=8, column=1)
tHigherSitNB_entry.insert(0, "100")  # Default value for Threat Nurturing Belonging Upper Bound

tk.Label(root, text="Ally Livelihood (Lower Bound)").grid(row=9, column=0)
aLowerSitL_entry = tk.Entry(root)
aLowerSitL_entry.grid(row=9, column=1)
aLowerSitL_entry.insert(0, "80")  # Default value for Ally Livelihood Lower Bound

tk.Label(root, text="Ally Livelihood (Upper Bound)").grid(row=10, column=0)
aHigherSitL_entry = tk.Entry(root)
aHigherSitL_entry.grid(row=10, column=1)
aHigherSitL_entry.insert(0, "100")  # Default value for Ally Livelihood Upper Bound

tk.Label(root, text="Ally Defensive Belonging (Lower Bound)").grid(row=11, column=0)
aLowerSitDB_entry = tk.Entry(root)
aLowerSitDB_entry.grid(row=11, column=1)
aLowerSitDB_entry.insert(0, "80")  # Default value for Ally Defensive Belonging Lower Bound

tk.Label(root, text="Ally Defensive Belonging (Upper Bound)").grid(row=12, column=0)
aHigherSitDB_entry = tk.Entry(root)
aHigherSitDB_entry.grid(row=12, column=1)
aHigherSitDB_entry.insert(0, "100")  # Default value for Ally Defensive Belonging Upper Bound

tk.Label(root, text="Ally Nurturing Belonging (Lower Bound)").grid(row=13, column=0)
aLowerSitNB_entry = tk.Entry(root)
aLowerSitNB_entry.grid(row=13, column=1)
aLowerSitNB_entry.insert(0, "80")  # Default value for Ally Nurturing Belonging Lower Bound

tk.Label(root, text="Ally Nurturing Belonging (Upper Bound)").grid(row=14, column=0)
aHigherSitNB_entry = tk.Entry(root)
aHigherSitNB_entry.grid(row=14, column=1)
aHigherSitNB_entry.insert(0, "100")  # Default value for Ally Nurturing Belonging Upper Bound

tk.Label(root, text="Prey Livelihood (Lower Bound)").grid(row=15, column=0)
pLowerSitL_entry = tk.Entry(root)
pLowerSitL_entry.grid(row=15, column=1)
pLowerSitL_entry.insert(0, "80")  # Default value for Ally Livelihood Lower Bound

tk.Label(root, text="Prey Livelihood (Upper Bound)").grid(row=16, column=0)
pHigherSitL_entry = tk.Entry(root)
pHigherSitL_entry.grid(row=16, column=1)
pHigherSitL_entry.insert(0, "100")  # Default value for Ally Livelihood Upper Bound

tk.Label(root, text="Prey Defensive Belonging (Lower Bound)").grid(row=17, column=0)
pLowerSitDB_entry = tk.Entry(root)
pLowerSitDB_entry.grid(row=17, column=1)
pLowerSitDB_entry.insert(0, "80")  # Default value for Ally Defensive Belonging Lower Bound

tk.Label(root, text="Prey Defensive Belonging (Upper Bound)").grid(row=18, column=0)
pHigherSitDB_entry = tk.Entry(root)
pHigherSitDB_entry.grid(row=18, column=1)
pHigherSitDB_entry.insert(0, "100")  # Default value for Ally Defensive Belonging Upper Bound

tk.Label(root, text="Prey Nurturing Belonging (Lower Bound)").grid(row=19, column=0)
pLowerSitNB_entry = tk.Entry(root)
pLowerSitNB_entry.grid(row=19, column=1)
pLowerSitNB_entry.insert(0, "80")  # Default value for Ally Nurturing Belonging Lower Bound

tk.Label(root, text="Prey Nurturing Belonging (Upper Bound)").grid(row=20, column=0)
pHigherSitNB_entry = tk.Entry(root)
pHigherSitNB_entry.grid(row=20, column=1)
pHigherSitNB_entry.insert(0, "100")  # Default value for Ally Nurturing Belonging Upper Bound

tk.Label(root, text="Society's Livelihood").grid(row=21, column=0)
societyL_entry = tk.Entry(root)
societyL_entry.grid(row=21, column=1)
societyL_entry.insert(0, "70") 

tk.Label(root, text="Society's Nurturing Belonging").grid(row=22, column=0)
societyNB_entry = tk.Entry(root)
societyNB_entry.grid(row=22, column=1)
societyNB_entry.insert(0, "70") 

tk.Label(root, text="Society's Defensive Belonging").grid(row=23, column=0)
societyDB_entry = tk.Entry(root)
societyDB_entry.grid(row=23, column=1)
societyDB_entry.insert(0, "70") 

# Add the new fields for Risk_Aversion, Risk_Threshold, MainB, Training_Episodes, LR
tk.Label(root, text="Risk Aversion (1 is neutral, <1 is inclined to risk, >1 is risk averse)").grid(row=24, column=0)
risk_aversion_entry = tk.Entry(root)
risk_aversion_entry.grid(row=24, column=1)
risk_aversion_entry.insert(0, "1.2")  # Default value for Risk Aversion

tk.Label(root, text="Risk Threshold (How risky an situation is to consider the risk. 10 = normal)").grid(row=25, column=0)
risk_threshold_entry = tk.Entry(root)
risk_threshold_entry.grid(row=25, column=1)
risk_threshold_entry.insert(0, "10")  # Default value for Risk Threshold

tk.Label(root, text="Reward Inclination (1 is neutral, <1 is reward averse, >1 is inclined to reward)").grid(row=26, column=0)
reward_inclination_entry = tk.Entry(root)
reward_inclination_entry.grid(row=26, column=1)
reward_inclination_entry.insert(0, "1")  

tk.Label(root, text="Reward Threshold (How rewarding a situation is to consider it. 0 = normal)").grid(row=27, column=0)
reward_threshold_entry = tk.Entry(root)
reward_threshold_entry.grid(row=27, column=1)
reward_threshold_entry.insert(0, "0")  # Default value for Risk Threshold

tk.Label(root, text="Learning Period (How many situations it will go through picking random actions)").grid(row=28, column=0)
learning_period_entry = tk.Entry(root)
learning_period_entry.grid(row=28, column=1)
learning_period_entry.insert(0, "600")  # Default value for Learning Rate (LR)

tk.Label(root, text="Training Episodes (How many times it encounters a situation including the learning period)").grid(row=29, column=0)
training_episodes_entry = tk.Entry(root)
training_episodes_entry.grid(row=29, column=1)
training_episodes_entry.insert(0, "1500")  # Default value for Training Episodes

tk.Label(root, text="Learning Rate (LR of the neural network)").grid(row=30, column=0)
lr_entry = tk.Entry(root)
lr_entry.grid(row=30, column=1)
lr_entry.insert(0, "0.001")  # Default value for Learning Rate (LR)

# Add a dropdown for MainB selection (NB or DB)
tk.Label(root, text="Agent's Belonging Type Selection (NB or DB)").grid(row=31, column=0)
mainB_var = tk.StringVar(root)
mainB_var.set("NB")  # Default value for MainB
mainB_dropdown = tk.OptionMenu(root, mainB_var, "NB", "DB")
mainB_dropdown.grid(row=31, column=1)

# Create a button to start the simulation
start_button = tk.Button(root, text="Start Simulation", command=start_simulation)
start_button.grid(row=32, column=0, columnspan=2)

tk.Label(root, text = "Model Name").grid(row = 33, column = 0)
model_name_entry = tk.Entry(root)
model_name_entry.grid(row = 33, column = 1)
model_name_entry.insert(0, "default_model")

def save_model():
    global agent
    if agent is None:
        agent = Agent(actions=list(Action), Lr=lr_entry.get(), Learning_Period=learning_period_entry.get())

    name = model_name_entry.get()
    if not name:
        messagebox.showerror("Error", "Please enter a model name to save.")
        return

    try:
        agent.save_models(name)

        # Build a dictionary to pass to save the stats
        stats_entries = {
            "prob_threat": prob_threat_entry,
            "prob_ally": prob_ally_entry,
            "prob_prey": prob_prey_entry,
            "tLowerSitL": tLowerSitL_entry,
            "tHigherSitL": tHigherSitL_entry,
            "tLowerSitDB": tLowerSitDB_entry,
            "tHigherSitDB": tHigherSitDB_entry,
            "tLowerSitNB": tLowerSitNB_entry,
            "tHigherSitNB": tHigherSitNB_entry,
            "aLowerSitL": aLowerSitL_entry,
            "aHigherSitL": aHigherSitL_entry,
            "aLowerSitDB": aLowerSitDB_entry,
            "aHigherSitDB": aHigherSitDB_entry,
            "aLowerSitNB": aLowerSitNB_entry,
            "aHigherSitNB": aHigherSitNB_entry,
            "pLowerSitL": pLowerSitL_entry,
            "pHigherSitL": pHigherSitL_entry,
            "pLowerSitDB": pLowerSitDB_entry,
            "pHigherSitDB": pHigherSitDB_entry,
            "pLowerSitNB": pLowerSitNB_entry,
            "pHigherSitNB": pHigherSitNB_entry,
            "societyL": societyL_entry,
            "societyNB": societyNB_entry,
            "societyDB": societyDB_entry,
            "Risk_Aversion": risk_aversion_entry,
            "Risk_Threshold": risk_threshold_entry,
            "Reward_Inclination": reward_inclination_entry,
            "Reward_Threshold": reward_threshold_entry,
            "MainB": mainB_var,  # This can be a string from dropdown
            "Training_Episodes": training_episodes_entry,
            "Learning_Period": learning_period_entry,
            "Learning_Rate": lr_entry
        }

        # Save stats to file
        agent.save_training_stats(stats_entries, folder_name=name)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save model: {str(e)}")


def load_model():
    name = model_name_entry.get()
    if not name:
        messagebox.showerror("Error", "Please enter a model name to load.")
        return
    try:
        global load_model_entry
        load_model_entry = not load_model_entry

        if load_model_entry:
            load_button.config(text="Unload Model")
            load_saved_stats(name)
            messagebox.showinfo("Success", f"Model '{name}' loaded successfully")
        else:
            load_button.config(text="Load Model")
            messagebox.showinfo("Success", f"Model '{name}' unloaded successfully")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")


save_button = tk.Button(root, text="Save Model", command=save_model)
save_button.grid(row=34, column=0)

load_button = tk.Button(root, text="Load Model", command=load_model)
load_button.grid(row=34, column=1)

def load_saved_stats(model_name):
    try:
        stats_path = os.path.join("saved_stats", model_name, "training_stats.json")
        if not os.path.exists(stats_path):
            messagebox.showwarning("Not Found", f"No saved stats found for model '{model_name}'.")
            return

        with open(stats_path, "r") as f:
            saved_stats = json.load(f)

        # Map keys to entry widgets and dropdown variable
        entry_mapping = {
            "prob_threat": prob_threat_entry,
            "prob_ally": prob_ally_entry,
            "prob_prey": prob_prey_entry,
            "tLowerSitL": tLowerSitL_entry,
            "tHigherSitL": tHigherSitL_entry,
            "tLowerSitDB": tLowerSitDB_entry,
            "tHigherSitDB": tHigherSitDB_entry,
            "tLowerSitNB": tLowerSitNB_entry,
            "tHigherSitNB": tHigherSitNB_entry,
            "aLowerSitL": aLowerSitL_entry,
            "aHigherSitL": aHigherSitL_entry,
            "aLowerSitDB": aLowerSitDB_entry,
            "aHigherSitDB": aHigherSitDB_entry,
            "aLowerSitNB": aLowerSitNB_entry,
            "aHigherSitNB": aHigherSitNB_entry,
            "pLowerSitL": pLowerSitL_entry,
            "pHigherSitL": pHigherSitL_entry,
            "pLowerSitDB": pLowerSitDB_entry,
            "pHigherSitDB": pHigherSitDB_entry,
            "pLowerSitNB": pLowerSitNB_entry,
            "pHigherSitNB": pHigherSitNB_entry,
            "societyL": societyL_entry,
            "societyNB": societyNB_entry,
            "societyDB": societyDB_entry,
            "Risk_Aversion": risk_aversion_entry,
            "Risk_Threshold": risk_threshold_entry,
            "Reward_Inclination": reward_inclination_entry,
            "Reward_Threshold": reward_threshold_entry,
            "Training_Episodes": training_episodes_entry,
            "Learning_Period": learning_period_entry,
            "Learning_Rate": lr_entry
        }

        # Populate entry fields
        for key, entry_widget in entry_mapping.items():
            if key in saved_stats:
                value = str(saved_stats[key])
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, value)

        # Set dropdown value
        if "MainB" in saved_stats:
            mainB_var.set(saved_stats["MainB"])

        messagebox.showinfo("Loaded", f"Successfully loaded stats for model '{model_name}'")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load stats: {str(e)}")

def predict_action():
    global predict_action_entry
    predict_action_entry = not predict_action_entry
    if predict_action_entry:
        predict_button.config(text = "Not Predict Action Mode")
    else:
        predict_button.config(text = "Predict Action Mode")

predict_button = tk.Button(root, text = "Predict Action Mode", command = predict_action)
predict_button.grid(row=35, column=0)


# Start the Tkinter event loop
root.mainloop()