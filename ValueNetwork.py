import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
class ValueNetwork(nn.Module):
    def __init__(self, n_states, n_actions,  fc1_dims = 128, fc2_dims = 64, alpha = 0.001):
        super(ValueNetwork,self).__init__()
        self.fc1 = nn.Linear(n_states, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.n_actions = n_actions
        self.softmax = nn.Softmax(dim = -1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        action_probs = self.softmax(x)
        return action_probs
    
    def choose_action(self, state):
        action_probs = self.forward(state)
        action_probs = Categorical(action_probs)
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def normalize_rewards(self, rewards, r_min=-100, r_max=100):
        # Normalize rewards based on expected range [r_min, r_max]
        normalized_rewards = 2 * (rewards - r_min) / (r_max - r_min) - 1
        # Ensure the normalized rewards are clipped between -1 and 1
        normalized_rewards = torch.clamp(normalized_rewards, -1, 1)
        return normalized_rewards


    
    def create_reward_tensor(self, actions, rewards, batch_size):
        reward_tensor = torch.zeros(batch_size, self.n_actions)
        for i, action in enumerate(actions):
            reward_tensor[i, action] = rewards[i] if rewards.dim() == 1 else rewards.item()
        return reward_tensor
    
    def learn(self, states, actions, rewards):
        batch_size = len(states)
        states = torch.tensor(states, dtype=torch.float32)
        #actions = self.one_hot_encode(actions)
        actions = torch.tensor(actions, dtype = torch.int64)
        rewards = torch.tensor(rewards, dtype = torch.float)
        
        # Normalize rewards between -1 and 1
        normalized_rewards = self.normalize_rewards(rewards)
        print("batch size: ", batch_size)
        print("Rewards: ", rewards)
        print("Rewards Size: ", rewards.size())
        print("Normalized rewards: ", normalized_rewards)
        # Create a tensor with rewards at action positions
        reward_tensor = self.create_reward_tensor(actions, normalized_rewards, batch_size)
        print("reward tensors: ", reward_tensor)
        action_probs = self.forward(states)
        print("action probs: ", action_probs)
        self.optimizer.zero_grad()
        # Calculate loss using MSE between predicted rewards and the reward tensor
        loss = self.criterion(action_probs, reward_tensor)
        
        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def one_hot_encode(self, action):
        one_hot = torch.zeros(3)
        one_hot[action] = 1
        return one_hot
