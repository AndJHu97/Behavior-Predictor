import tkinter as tk
from tkinter import messagebox
import random
from situations import Action, SituationType, Threat, Ally, Prey
from agent import Character, Agent
from helper import plot_curves

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
    MainB = mainB_var.get()  # Get the value from the dropdown menu
    Training_Episodes = int(training_episodes_entry.get())
    Learning_Period = int(learning_period_entry.get())
    Lr = float(lr_entry.get())
    
    # Call the main function with these values
    main(prob_threat, prob_ally, prob_prey, tLowerSitL, tHigherSitL, tLowerSitDB, tHigherSitDB, tLowerSitNB, tHigherSitNB,
         aLowerSitL, aHigherSitL, aLowerSitDB, aHigherSitDB, aLowerSitNB, aHigherSitNB,
         pLowerSitL, pHigherSitL, pLowerSitDB, pHigherSitDB, pLowerSitNB, pHigherSitNB,
         societyL, societyNB, societyDB,
         Risk_Aversion, Risk_Threshold, MainB, Training_Episodes, Lr, Learning_Period)

def main(prob_threat, prob_ally, prob_prey, tLowerSitL, tHigherSitL, tLowerSitDB, tHigherDB, tLowerSitNB, tHigherSitNB, 
         aLowerSitL, aHigherSitL, aLowerSitDB, aHigherSitDB, aLowerSitNB, aHigherSitNB, 
         pLowerSitL, pHigherSitL, pLowerSitDB, pHigherSitDB, pLowerSitNB, pHigherSitNB,
         societyL, societyNB, societyDB,
         Risk_Aversion, Risk_Threshold, MainB, Training_Episodes, LR, Learning_Period):
    # Define the character and the initial situation
    absL = 100
    absNB = 100
    absDB = 100
    tSitL = random.randint(tLowerSitL, tHigherSitL)
    tSitDB = random.randint(tLowerSitDB, tHigherDB)
    tSitNB = random.randint(tLowerSitNB, tHigherSitNB)
    aSitL = random.randint(aLowerSitL, aHigherSitL)
    aSitDB = random.randint(aLowerSitDB, aHigherSitDB)
    aSitNB = random.randint(aLowerSitNB, aHigherSitNB)
    pSitL = random.randint(pLowerSitL, pHigherSitL)
    pSitDB = random.randint(pLowerSitDB, pHigherSitDB)
    pSitNB = random.randint(pLowerSitNB, pHigherSitNB)

    character = Character(risk_aversion= Risk_Aversion, risk_threshold=Risk_Threshold, absL=absL, absNB=absNB, absDB = absDB, mainB = MainB)
    randomChance = random.random()
    if randomChance < prob_threat:
        situation = Threat(sitL=tSitL, sitDB=tSitDB, sitNB = tSitNB, sitType=SituationType.Threat, societyL=societyL, societyDB=societyDB, societyNB=societyNB)
    elif randomChance < prob_threat + prob_ally:
        situation = Ally(sitL = aSitL, sitDB= aSitDB, sitNB = aSitNB, sitType=SituationType.Ally)
    else:
        situation = Prey(sitL = pSitL, sitDB= pSitDB, sitNB = pSitNB, sitType=SituationType.Prey)
    agent = Agent(actions=list(Action), Lr=LR, Learning_Period= Learning_Period)
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
    rounds_encountered = 0
    for episode in range(Training_Episodes):
        state = agent.get_state(character, situation)
        action = agent.select_action(character, state, rounds_encountered)

        relL_values.append(character.relL)
        relNB_values.append(character.relNB)
        relDB_values.append(character.relDB)
        sitL_values.append(situation.sitL)
        #Change to right situation 
        sitNB_values.append(situation.sitNB) 
        sitDB_values.append(situation.sitDB)
        sit_types.append(situation.sitType.value)
        action_values.append(action)
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
       
        if death:
            print(f"Character died after {survival_rounds} rounds")
            character = Character(risk_aversion= Risk_Aversion, risk_threshold= Risk_Threshold,  absL=absL, absNB=absNB, absDB = absDB, mainB = MainB)
            #not long term because not very human like
            #blStore = agent.train_long_memory()
            #lLoss_values.append(blStore[0])
            #bLoss_values.append(blStore[1])
        tSitL = random.randint(tLowerSitL, tHigherSitL)
        tSitDB = random.randint(tLowerSitDB, tHigherDB)
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

    plot_curves(relL_values, relDB_values, sitL_values, sitDB_values, action_values, survival_rounds_values, lLoss_values, dbLoss_values, sit_types)

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

tk.Label(root, text="Learning Period (How many situations it will go through picking random actions)").grid(row=26, column=0)
learning_period_entry = tk.Entry(root)
learning_period_entry.grid(row=26, column=1)
learning_period_entry.insert(0, "600")  # Default value for Learning Rate (LR)

tk.Label(root, text="Training Episodes (How many times it encounters a situation including the learning period)").grid(row=27, column=0)
training_episodes_entry = tk.Entry(root)
training_episodes_entry.grid(row=27, column=1)
training_episodes_entry.insert(0, "1500")  # Default value for Training Episodes

tk.Label(root, text="Learning Rate (LR of the neural network)").grid(row=28, column=0)
lr_entry = tk.Entry(root)
lr_entry.grid(row=28, column=1)
lr_entry.insert(0, "0.001")  # Default value for Learning Rate (LR)

# Add a dropdown for MainB selection (NB or DB)
tk.Label(root, text="Agent's Belonging Type Selection (NB or DB)").grid(row=29, column=0)
mainB_var = tk.StringVar(root)
mainB_var.set("NB")  # Default value for MainB
mainB_dropdown = tk.OptionMenu(root, mainB_var, "NB", "DB")
mainB_dropdown.grid(row=29, column=1)

# Create a button to start the simulation
start_button = tk.Button(root, text="Start Simulation", command=start_simulation)
start_button.grid(row=30, column=0, columnspan=2)

# Start the Tkinter event loop
root.mainloop()