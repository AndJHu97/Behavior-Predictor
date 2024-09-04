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
        self.lFightModel = ValueNetwork(6, 1, alpha = LR)
        self.bFightModel = ValueNetwork(6, 1, alpha = LR)
        self.lFleeModel = ValueNetwork(6, 1, alpha = LR)
        self.bFleeModel = ValueNetwork(6, 1, alpha = LR)
        self.lSelectedActionModel = None
        self.bSelectedActionModel = None
        self.selectedActionName = ""
    def select_action(self,state, round_survived):
        # Implementing epsilon-greedy policy
        if random.randint(0,100) < self.epsilon - round_survived:
            move = random.choice(self.actions)
            if move == Action.Fight:
                self.lSelectedActionModel = self.lFightModel
                self.bSelectedActionModel = self.bFightModel
            elif move == Action.Flee:
                self.lSelectedActionModel = self.lFleeModel
                self.bSelectedActionModel = self.bFleeModel
            return move.value
        else:
            state0 = torch.tensor(state, dtype = torch.float) #Convert to tensor
            lFightPrediction = self.lFightModel(state0) #Execute forward function
            bFightPrediction = self.bFightModel(state0)
            lFleePrediction = self.lFleeModel(state0)
            bFleePrediction = self.bFleeModel(state0)
            #choose the one that is the most pressing to do
           
           # Calculate the sum of predictions for fight and flee
            fight_sum = lFightPrediction + bFightPrediction
            flee_sum = lFleePrediction + bFleePrediction
            print("fight_sum: ", fight_sum)
            print("flee_sum: ", flee_sum)
            # Determine which action has the maximum sum
            if fight_sum > flee_sum:
                max_name = "Fight"
                max_value = fight_sum
                self.lSelectedActionModel = self.lFightModel
                self.bSelectedActionModel = self.bFightModel
                move = 0  # Fight
            else:
                max_name = "Flee"
                max_value = flee_sum
                self.lSelectedActionModel = self.lFleeModel
                self.bSelectedActionModel = self.bFleeModel
                move = 1  # Flee

            self.selectedActionName = max_name
            return move
    def remember(self, state, action, lReward, bReward, lSelectedActionModel, bSelectedActionModel):
        #append a tuple
        self.memory.append((state, [action], lReward, bReward, lSelectedActionModel, bSelectedActionModel)) 
    def train_long_memory(self): #train with whole batch
        if len(self.memory) > BATCH_SIZE: #If above the batch size, randomly grabs samples used in training iteration
            mini_sample = random.sample(self.memory, BATCH_SIZE) # randomly samples tuples
        else:
            mini_sample = self.memory

        #unzip will give it batchsize
        states, actions, lRewards, bReward, lModel, bModel = zip(*mini_sample) #unzips
        lLoss = lModel.learn(states, actions, lRewards)
        bLoss = bModel.learn(states, actions, bReward)
        return [lLoss, bLoss]
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)
    def train_short_memory(self, state, action, lReward, bReward):
        #Should learn from lReward for lAction and bReward from bAction
        #Could use the self.selectedActionName to pick the right things to learn from
        lLoss = self.lSelectedActionModel.learn(state, [action], lReward)
        bLoss = self.bSelectedActionModel.learn(state, [action], bReward)
        return [lLoss, bLoss]
    def get_state(self, character, situation):
        state_features = [
            character.relL,
            character.relB,
            situation.sitL,
            situation.sitB
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
    absB = 100
    tSitL = random.randint(40, 90)
    tSitB = random.randint(40, 90)
    aSitL = random.randint(40, 90)
    aSitB = random.randint(40, 90)
    prob_threat = 1
    prob_ally = 1 - prob_threat
    character = Character(absL=absL, absB=absB)
    if random.random() < prob_threat:
        situation = Threat(sitL=tSitL, sitB=tSitB, sitType=SituationType.Threat)
    else:
        situation = Ally(sitL = aSitL, sitB= aSitB, sitType=SituationType.Ally)
    agent = Agent(actions=list(Action))
    relL_values = []
    relB_values = []
    sitL_values = []
    sitB_values = []
    sit_types = []
    action_values = []
    survival_rounds_values = []
    rewards_values = []
    bLoss_values = []
    lLoss_values = []

    for episode in range(1000):
        state = agent.get_state(character, situation)
        action = agent.select_action(state, character.survival_rounds)

        relL_values.append(character.relL)
        relB_values.append(character.relB)
        sitL_values.append(situation.sitL)
        sitB_values.append(situation.sitB)
        sit_types.append(situation.sitType.value)
        action_values.append(action)
        blStore = []
        lReward, bReward, death, survival_rounds = situation.process_action(character, action)
        survival_rounds_values.append(survival_rounds)
        rewards_values.append(bReward)
        blStore = agent.train_short_memory(state, action, lReward, bReward)
        lLoss_values.append(blStore[0])
        bLoss_values.append(blStore[1])
        agent.remember(state,action,lReward, bReward, agent.lSelectedActionModel, agent.bSelectedActionModel)
        character.set_stats(character.absL + 5, character.absB + 5)
       
        if death:
            print(f"Character died after {survival_rounds} rounds")
            character = Character(absL=absL, absB=absB)
            blStore = agent.train_long_memory()
            lLoss_values.append(blStore[0])
            bLoss_values.append(blStore[1])
        tSitL = random.randint(40, 90)
        tSitB = random.randint(40, 90)
        aSitL = random.randint(40, 90)
        aSitB = random.randint(40, 90)
        #character = Character(absL=absL, absB=absB)
        if random.random() < prob_threat:
            situation = Threat(sitL=tSitL, sitB=tSitB, sitType=SituationType.Threat)
        else:
            situation = Ally(sitL = aSitL, sitB= aSitB, sitType=SituationType.Ally)

    plot_curves(relL_values, relB_values, sitL_values, sitB_values, action_values, survival_rounds_values, lLoss_values, bLoss_values, sit_types)

if __name__ == "__main__":
    main()