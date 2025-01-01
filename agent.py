import random
from enum import Enum
from situations import Action, SituationType
#from PGmodel import PolicyNetwork, PolicyTrainer
from ValueNetwork import ValueNetwork
from collections import deque, defaultdict
import numpy as np
import torch


MAX_MEMORY = 100_000
BATCH_SIZE = 100
LR = 0.001
LEARNING_PERIOD = 600
TRAINING_EPISODES = 1500
RISK_THRESHOLD = 20
RISK_AVERSION = 1

class Character:
    #need to get rid of abs, I stopped using them
    def __init__(self, risk_aversion=1.5, risk_threshold=5, absL=100, absDB=100, absNB=100, mainB="DB"):
        #multiplier for risky actions (negative rewards)
        self.risk_aversion = risk_aversion
        #cut off of negativity to care about
        self.risk_cutoff = risk_threshold
        self.absL = absL
        self.absNB = absNB
        self.absDB = absDB
        self.mainB = mainB
        self.relL = self.calculate_relative_stats(absL)
        self.relNB = self.calculate_relative_stats(absNB)
        self.relDB = self.calculate_relative_stats(absDB)
        self.survival_rounds = 0
    
    #This doesn't work anymore because I'm using rel now
    def set_stats(self, relL, relDB, relNB):
        self.relL = relL
        self.relNB = relNB
        self.relDB = relDB

    def mainRelB(self):
        if self.mainB == "DB":
            return self.relDB
        elif self.mainB == "NB":
            return self.relNB
        else:
            raise ValueError(f"Invalid value for mainB: {self.mainB}")

    #this will mimic real life more. It's harder to lose when are close to dying and harder to gain when almost full health. 
    def calculate_relative_stats(self, stat):
        return modified_exponential_bound(stat)

def modified_exponential_bound(x, k=0.02):
    return 100 * (1 - np.exp(-k * x))

class Agent:
    def __init__(self, actions, Lr, Learning_Period):
        self.actions = actions
        #self.epsilon = epsilon
        self.memory = deque(maxlen = MAX_MEMORY) #When exceed memory, will remove memory/popleft
        self.lFightModel = ValueNetwork(8, 1, alpha = Lr)
        self.nbFightModel = ValueNetwork(8, 1, alpha = Lr)
        self.dbFightModel = ValueNetwork(8, 1, alpha = Lr)
        self.lFleeModel = ValueNetwork(8, 1, alpha = Lr)
        self.nbFleeModel = ValueNetwork(8, 1, alpha = Lr)
        self.dbFleeModel = ValueNetwork(8, 1, alpha = Lr)
        self.lBefriendModel = ValueNetwork(8, 1, alpha = Lr)
        self.nbBefriendModel = ValueNetwork(8, 1, alpha = Lr)
        self.dbBefriendModel = ValueNetwork(8, 1, alpha = Lr)
        self.Learning_Period = Learning_Period
        self.lSelectedActionModel = None
        self.dbSelectedActionModel = None
        self.nbSelectedActionModel = None
        self.selectedActionName = ""
        self.selectedBType = ""
    
    #select actions and save the l and b models being used
    def select_action(self, character, state, rounds_encountered):
    # Implementing epsilon-greedy policy
        if random.randint(0, self.Learning_Period) < self.Learning_Period - rounds_encountered:
            # Exploration: Random action
            move = random.choice(self.actions)
            if move == Action.Fight:
                self.set_selected_models("Fight")
            elif move == Action.Flee:
                self.set_selected_models("Flee")
            elif move == Action.Befriend:
                self.set_selected_models("Befriend")
            return move.value
        else:
            # Exploitation: Evaluate models
            state_tensor = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
            predictions = {
                "Fight": {
                    "L": self.lFightModel(state_tensor),
                    "NB": self.nbFightModel(state_tensor),
                    "DB": self.dbFightModel(state_tensor)
                },
                "Flee": {
                    "L": self.lFleeModel(state_tensor),
                    "NB": self.nbFleeModel(state_tensor),
                    "DB": self.dbFleeModel(state_tensor)
                },
                "Befriend": {
                    "L": self.lBefriendModel(state_tensor),
                    "NB": self.nbBefriendModel(state_tensor),
                    "DB": self.dbBefriendModel(state_tensor)
                }
            }

            # Flatten the predictions into a list of tuples: (action, model_type, reward)
            flattened_predictions = [
                ("Fight", "L", predictions["Fight"]["L"]),
                ("Fight", "NB", predictions["Fight"]["NB"]),
                ("Fight", "DB", predictions["Fight"]["DB"]),
                ("Flee", "L", predictions["Flee"]["L"]),
                ("Flee", "NB", predictions["Flee"]["NB"]),
                ("Flee", "DB", predictions["Flee"]["DB"]),
                ("Befriend", "L", predictions["Befriend"]["L"]),
                ("Befriend", "NB", predictions["Befriend"]["NB"]),
                ("Befriend", "DB", predictions["Befriend"]["DB"]),
            ]

            #process to select out too risky 
            #group predictions by action
            risk_avoidance_grouped_predictions = defaultdict(list)
            for action, model_type, reward in flattened_predictions:
                if model_type == "L" or model_type == character.mainB:
                    risk_avoidance_grouped_predictions[action].append(reward)

            # these are actions to remove
            risky_actions_to_avoid = set()

            #look through each actions. If the risky rewards * risk_aversion is greater than the reward, then avoid
            for action, rewards in risk_avoidance_grouped_predictions.items():
                max_action_positive = max((r for r in rewards if r>0), default= 0)
                print("max action positive: ", max_action_positive, " for action: ", action)
                for reward in rewards:
                    print("reward: ", reward.item(), " for action: ", action)
                    if reward < 0:
                        adjusted_risk = reward * character.risk_aversion
                        #print("adjusted risk: ", adjusted_risk)
                        #if adjusted risk is worse than any positive rewards and if the result is above the cutoff to caring about the risk 
                        if abs(adjusted_risk) > max_action_positive and abs(adjusted_risk) > character.risk_cutoff:
                            risky_actions_to_avoid.add(action)
                            print("Risky actions to remove: ", action)
                            break #done because this action is already considered too risky

            # Filter to only consider models relevant to character.mainB and L
            # Ignore B not consistent with mainB
            relevant_predictions = [
                (action, model_type, reward)
                for action, model_type, reward in flattened_predictions
                if model_type in ["L", character.mainB] and action not in risky_actions_to_avoid
            ]

           #for action, model_type, reward in relevant_predictions:
            #   print(f"Action: {action}, Model: {model_type}, Reward: {reward}")

            if len(relevant_predictions) > 0:
                # Find the maximum reward and corresponding action
                best_action, max_model, max_reward = max(relevant_predictions, key=lambda x: x[2])
                print(f"Best action: {best_action}, Max reward: {max_reward.item()}, Model type: {max_model}")
            else:
                best_action = "Depression"
                

            #print(f"Character mainB: {character.mainB}")
            #print(f"Relevant predictions: {relevant_predictions}")
           

            # Set selected models based on the chosen action
            self.set_selected_models(best_action)
            return self.return_action_number(best_action)

    def return_action_number(self, action_type):
        if action_type == "Fight":
            return 0
        elif action_type == "Flee":
            return 1
        elif action_type == "Befriend":
            return 2
        return -1 #depression or learned helplessness
        
    def set_selected_models(self, action_type):
        if action_type == "Fight":
            self.lSelectedActionModel = self.lFightModel
            self.nbSelectedActionModel = self.nbFightModel
            self.dbSelectedActionModel = self.dbFightModel
        elif action_type == "Flee":
            self.lSelectedActionModel = self.lFleeModel
            self.nbSelectedActionModel = self.nbFleeModel
            self.dbSelectedActionModel = self.dbFleeModel
        elif action_type == "Befriend":
            self.lSelectedActionModel = self.lBefriendModel
            self.nbSelectedActionModel = self.nbBefriendModel
            self.dbSelectedActionModel = self.dbBefriendModel
        elif action_type == "Depression":
            self.lSelectedActionModel = None
            self.nbSelectedActionModel = None
            self.dbSelectedActionModel = None


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
        if self.lSelectedActionModel != None:
            lLoss = self.lSelectedActionModel.learn(state, [action], lReward)
            dbLoss = self.dbSelectedActionModel.learn(state, [action], dbReward)
            nbLoss = self.nbSelectedActionModel.learn(state, [action], nbReward)
        else:
            lLoss = 0
            dbLoss = 0
            nbLoss = 0

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

