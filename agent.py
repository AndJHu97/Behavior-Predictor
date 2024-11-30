import random
from enum import Enum
from simulation import Action, SituationType, Character, Situation, Threat, Ally
#from PGmodel import PolicyNetwork, PolicyTrainer
from ValueNetwork import ValueNetwork
from collections import deque
import numpy as np
import torch
from helper import plot_curves

MAX_MEMORY = 100_000
BATCH_SIZE = 100
LR = 0.001

class Agent:
    def __init__(self, actions, epsilon = 80):
        self.actions = actions
        self.epsilon = epsilon
        self.memory = deque(maxlen = MAX_MEMORY) #When exceed memory, will remove memory/popleft
        self.lFightModel = ValueNetwork(8, 1, alpha = LR)
        self.nbFightModel = ValueNetwork(8, 1, alpha = LR)
        self.dbFightModel = ValueNetwork(8, 1, alpha = LR)
        self.lFleeModel = ValueNetwork(8, 1, alpha = LR)
        self.nbFleeModel = ValueNetwork(8, 1, alpha = LR)
        self.dbFleeModel = ValueNetwork(8, 1, alpha = LR)
        self.lSelectedActionModel = None
        self.dbSelectedActionModel = None
        self.nbSelectedActionModel = None
        self.selectedActionName = ""
        self.selectedBType = ""
    
    #select actions and save the l and b models being used
    def select_action(self, state, round_survived):
        # Implementing epsilon-greedy policy
        #NEED TO FIX
        if random.randint(0,100) < self.epsilon - round_survived:
            move = random.choice(self.actions)
            if move == Action.Fight:
                self.lSelectedActionModel = self.lFightModel
                self.dbSelectedActionModel = self.dbFightModel
                self.nbSelectedActionModel = self.nbFightModel
            elif move == Action.Flee:
                self.lSelectedActionModel = self.lFleeModel
                self.dbSelectedActionModel = self.dbFleeModel
                self.nbSelectedActionModel = self.nbFleeModel
            return move.value
        else:
            state0 = torch.tensor(state, dtype = torch.float) #Convert to tensor
            lFightPrediction = self.lFightModel(state0) #Execute forward function
            nbFightPrediction = self.nbFightModel(state0)
            dbFightPrediction = self.dbFightModel(state0)
            lFleePrediction = self.lFleeModel(state0)
            nbFleePrediction = self.nbFleeModel(state0)
            dbFleePrediction = self.dbFleeModel(state0)
            #choose the one that is the most pressing to do
            # Store predictions with labels
            actionPredictions = {
                "lFightModel": lFightPrediction,
                "nbFightModel": nbFightPrediction,
                "dbFightModel": dbFightPrediction,
                "lFleeModel" : lFleePrediction,
                "nbFleeModel" : nbFleePrediction,
                "dbFleeModel" : dbFleePrediction
            }
           # Find the max prediction and its corresponding model
            max_model, max_prediction = max(actionPredictions.items(), key=lambda x: x[1])
            print("max model: ", max_model)
            print("max prediction: ", max_prediction)
            # Determine which action has the maximum sum
            if max_model == "lFightModel":
                self.lSelectedActionModel = self.lFightModel
                self.nbSelectedActionModel = self.nbFightModel
                self.dbSelectedActionModel = self.dbFightModel
                move = 0 #Fight
                self.selectedBType = ""
            elif max_model == "nbFightModel":
                self.lSelectedActionModel = self.lFightModel
                self.nbSelectedActionModel = self.nbFightModel
                self.dbSelectedActionModel = self.dbFightModel
                move = 0
                self.selectedBType = "NB"
            elif max_model == "dbFightModel":
                self.lSelectedActionModel = self.lFightModel
                self.nbSelectedActionModel = self.nbFightModel
                self.dbSelectedActionModel = self.dbFightModel
                move = 0
                self.selectedBType = "DB"
            elif max_model == "lFleeModel":
                self.lSelectedActionModel = self.lFleeModel
                self.nbSelectedActionModel = self.nbFleeModel
                self.dbSelectedActionModel = self.dbFleeModel
                move = 1
                self.selectedBType = ""
            elif max_model == "nbFleeModel":
                self.lSelectedActionModel = self.lFleeModel
                self.nbSelectedActionModel = self.nbFleeModel
                self.dbSelectedActionModel = self.dbFleeModel
                move = 1
                self.selectedBType = "NB"
            elif max_model == "dbFleeModel":
                self.lSelectedActionModel = self.lFleeModel
                self.nbSelectedActionModel = self.nbFleeModel
                self.dbSelectedActionModel = self.dbFleeModel
                move = 1
                self.selectedBType = "DB"

            self.selectedActionName = max_model
            return move
    def remember(self, state, action, lReward, dbReward, nbReward, lSelectedActionModel, dbSelectedActionModel, nbSelectedActionModel):
        #append a tuple
        self.memory.append((state, [action], lReward, dbReward, nbReward, lSelectedActionModel, dbSelectedActionModel, nbSelectedActionModel)) 

    #Not using this anymore because not really representative of human learning
    def train_long_memory(self): #train with whole batch
        if len(self.memory) > BATCH_SIZE: #If above the batch size, randomly grabs samples used in training iteration
            mini_sample = random.sample(self.memory, BATCH_SIZE) # randomly samples tuples
        else:
            mini_sample = self.memory

        #unzip will give it batchsize
        states, actions, lRewards, dbRewards, nbRewards, lModels, dbModels, nbModels = zip(*mini_sample)
         # Check what lModels and bModels actually contain
        print(type(lModels[0]))  # Should be <class 'ValueNetwork'>
        print(type(dbModels[0]))  # Should be <class 'ValueNetwork'>

        # Assuming lModels and bModels are lists of ValueNetwork instances
        # This doesn't work because it is training all the actions for all models. Need to filter out but currently not using long term training
        lLoss = sum(model.learn(states, actions, lRewards) for model in lModels)
        dbLoss = sum(model.learn(states, actions, dbRewards) for model in dbModels)
        nbLoss = sum(model.learn(states, actions, nbRewards) for model in nbModels)
        return [lLoss, dbLoss, nbLoss]
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
    def train_short_memory(self, state, action, lReward, dbReward, nbReward):
        #Should learn from lReward for lAction and bReward from bAction
        #Could use the self.selectedActionName to pick the right things to learn from

        """
        if action == Action.Fight.value:
            lLoss = self.lFightModel.learn(state, [action], lReward)
            bLoss = self.nbFightModel.learn(state, [action], bReward)
        elif action == Action.Flee.value:
            lLoss = self.lFleeModel.learn(state, [action], lReward)
            bLoss = self.nbFleeModel.learn(state, [action], bReward)
        """

        lLoss = self.lSelectedActionModel.learn(state, [action], lReward)
        dbLoss = self.dbSelectedActionModel.learn(state, [action], dbReward)
        nbLoss = self.nbSelectedActionModel.learn(state, [action], nbReward)

        return [lLoss, dbLoss, nbLoss]
    def get_state(self, character, situation):
        state_features = [
            character.relL,
            character.relDB,
            character.relNB,
            situation.sitL,
            situation.sitDB,
            situation.sitNB
           # character.relL - situation.sitL,
            #character.relB - situation.sitB
        ]
        # Reshape state_features to have a single sample with multiple features
        #1: one row, -1: automatically calculate appropriate columns based on array original dim
        #(1,6)
        state_features = np.array(state_features)
        #max-min normalization (values between 0 and 1 to complement the one-hot encoded values of situation type)
        state_min = np.min(state_features)
        state_max = np.max(state_features)
        state_normalized = (state_features - state_min) / (state_max - state_min)
        sitType = self.integer_to_one_hot(situation.sitType.value, SituationType.Count.value)

        sitType_array = np.array(sitType)
        state = np.concatenate((state_normalized, sitType_array))
        return state

    def integer_to_one_hot(self, index, num_classes):
        """Converts an integer index to a one-hot encoded array."""
        one_hot = [0] * num_classes
        one_hot[index] = 1
        return one_hot

def main():
    # Define the character and the initial situation
    absL = 100
    absNB = 100
    absDB = 100
    tSitL = random.randint(40, 110)
    tSitDB = random.randint(60, 100)
    tSitNB = random.randint(30, 60)
    aSitL = random.randint(30, 60)
    aSitDB = random.randint(30, 60)
    aSitNB = random.randint(30, 60)
    prob_threat = 1
    prob_ally = 1 - prob_threat
    character = Character(absL=absL, absNB=absNB, absDB = absDB, mainB = "DB")
    if random.random() < prob_threat:
        situation = Threat(sitL=tSitL, sitDB=tSitDB, sitNB = tSitNB, sitType=SituationType.Threat)
    else:
        situation = Ally(sitL = aSitL, sitDB= aSitDB, sitNB = aSitNB, sitType=SituationType.Ally)
    agent = Agent(actions=list(Action))
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

    for episode in range(1000):
        state = agent.get_state(character, situation)
        action = agent.select_action(state, character.survival_rounds)

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
        character.set_stats(character.absL + 5, character.absDB + 5, character.absNB + 5)
       
        if death:
            print(f"Character died after {survival_rounds} rounds")
            character = Character(absL=absL, absNB=absNB, absDB = absDB, mainB = "DB")
            #not long term because not very human like
            #blStore = agent.train_long_memory()
            #lLoss_values.append(blStore[0])
            #bLoss_values.append(blStore[1])
        tSitL = random.randint(40, 110)
        tSitDB = random.randint(60, 100)
        tSitNB = random.randint(30, 60)
        aSitL = random.randint(30, 60)
        aSitDB = random.randint(30, 60)
        aSitNB = random.randint(30, 60)
        #character = Character(absL=absL, absB=absB)
        if random.random() < prob_threat:
            situation = Threat(sitL=tSitL, sitDB =tSitDB, sitNB = tSitNB, sitType=SituationType.Threat)
        else:
            situation = Ally(sitL = aSitL, sitDB = aSitDB, sitNB = aSitNB, sitType=SituationType.Ally)

    plot_curves(relL_values, relDB_values, sitL_values, sitDB_values, action_values, survival_rounds_values, lLoss_values, dbLoss_values, sit_types)

if __name__ == "__main__":
    main()