import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
#Contextual Bandit - Not used
class BanditNetwork(nn.Module):
    def __init__(self, n_states, n_actions,  fc1_dims = 128, fc2_dims = 64, alpha = 0.001):
        super(BanditNetwork,self).__init__()
        self.fc1 = nn.Linear(n_states, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.softmax = nn.Softmax(dim = -1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,state):
        state = torch.relu(self.fc1(state))
        state = torch.relu(self.fc2(state))
        action_probs = self.fc3(state)
        #action_probs = self.softmax(state)
        return action_probs
    
    def choose_action(self, state):
        action_probs = self.forward(state)
        action_probs = Categorical(action_probs)
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def learn(self, states, actions, rewards):
        states = torch.tensor(states, dtype=torch.float32)
        #actions = self.one_hot_encode(actions)
        actions = torch.tensor(actions, dtype = torch.int64)
        rewards = torch.tensor(rewards, dtype = torch.float)
        '''
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            #entering as just a number so need to do twice
            actions = torch.unsqueeze(actions,0)
            rewards = torch.unsqueeze(rewards, 0)
        '''
        action_probs = self.forward(states)
        if actions.dim() > 1:
            actions = actions.squeeze()
        '''
        # Ensure action_probs has the correct shape
        if action_probs.dim() == 1:
            action_probs = action_probs.unsqueeze(0)  # Convert to shape [1, num_classes]

        # Ensure that actions is 1D and get rid of batch dim. Only want the actions indices
        if actions.dim() > 1:
            actions = actions.squeeze()  # Remove any extra dimensions if necessary
        print("actions: ", actions)
        print("action_probs: ", action_probs.shape)
        self.optimizer.zero_grad()
        #Crossentropy calculates based on action_probs and correct indices of action by target tensor (actions)
        loss = self.criterion(action_probs, actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        '''
        self.optimizer.zero_grad()
        #Crossentropy calculates based on action_probs and correct indices of action by target tensor (actions)
        # Get predicted rewards for each action
        predicted_rewards = self.forward(states)
        
        # Calculate loss for the chosen actions
        loss = self.criterion(predicted_rewards.gather(1, actions.unsqueeze(1)).squeeze(), rewards)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def one_hot_encode(self, action):
        one_hot = torch.zeros(3)
        one_hot[action] = 1
        return one_hot
