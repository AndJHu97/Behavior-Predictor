import torch
import torch.nn as nn
import torch.optim as optim

class ValueNetwork(nn.Module):
    def __init__(self, n_states, n_actions,  fc1_dims = 128, fc2_dims = 64, alpha = 0.001):
        super(ValueNetwork,self).__init__()
        self.fc1 = nn.Linear(n_states, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.n_actions = n_actions
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #No soft max because this is value prediction
        x = self.fc3(x)
        return x
    
    def predict_value(self, state):
        """Predict the value for a given state"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            value = self.forward(state_tensor)
        return value.item()
    
    def normalize_rewards(self, rewards, r_min=-100, r_max=100):
        # Normalize rewards based on expected range [r_min, r_max]
        normalized_rewards = 2 * (rewards - r_min) / (r_max - r_min) - 1
        # Ensure the normalized rewards are clipped between -1 and 1
        normalized_rewards = torch.clamp(normalized_rewards, -1, 1)
        return normalized_rewards


    
    def create_reward_tensor(self, actions, rewards, batch_size):
        """This method is not used since we have separate networks per action"""
        reward_tensor = torch.zeros(batch_size, self.n_actions)
        for i, action in enumerate(actions):
            reward_tensor[i, action] = rewards[i] if rewards.dim() == 1 else rewards.item()
        return reward_tensor
    
    #Could delete actions as parameter
    def learn(self, states, actions, rewards):
        """
        Train the value network to predict rewards for this specific action
        Note: actions parameter is not used since each network represents one action
        
        Args:
            states: Single state [9] or batch of states [batch_size, 9]
            actions: Not used (kept for API compatibility)
            rewards: Single scalar, [reward], or [batch_size] of rewards
        """
        # Convert inputs to tensors
        states = torch.tensor(states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Normalize states to [batch_size, n_features]
        if states.dim() == 1:
            states = states.unsqueeze(0)  # [9] -> [1, 9]
        elif states.dim() != 2:
            raise ValueError(f"Expected states to be 1D or 2D, got {states.dim()}D")
        
        #This only does really batch size of 1
        if rewards.dim() == 0:  # Single scalar: 5.0
            rewards = rewards.unsqueeze(0).unsqueeze(0)  # [] -> [1, 1]
        elif rewards.dim() == 1:  # Either [reward] or [batch_size]. So iffy on this because could be reward, 1 rather than batch size, 1
            rewards = rewards.unsqueeze(1)  # [n] -> [n, 1]
        elif rewards.dim() == 2 and rewards.shape[1] == 1:
            pass  # Already [batch_size, 1]
        else:
            raise ValueError(f"Expected rewards shape [batch], [1], or [batch, 1], got {rewards.shape}")
        
        # Verify batch sizes match
        if states.shape[0] != rewards.shape[0]:
            raise ValueError(f"Batch size mismatch: states {states.shape[0]} vs rewards {rewards.shape[0]}")
        
        # Forward pass
        predicted_values = self.forward(states)
        
        # Calculate loss
        loss = self.criterion(predicted_values, rewards)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    #Don't use anymore since each network is for one action
    def one_hot_encode(self, action):
        one_hot = torch.zeros(3)
        one_hot[action] = 1
        return one_hot
