import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        #dim = -1 last, dim = 0 first dimension, dim = 1 second, etc.
        #Shape is (1,3) or (,3) for output
        #dim = 1 for the 3 part of (1,3) shape or dim = 0 for (3) w/o batch size for 3
        self.softmax0 = nn.Softmax(dim = 0)
        self.softmax1 = nn.Softmax(dim = 1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        if(len(x.shape) == 1):
            x = self.softmax0(x)
        else:
            x = self.softmax1(x)
        return x
    def save(self, file_name='policy_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path): #check if exists
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class PolicyTrainer:
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
    def train_step(self, states, actions, rewards):
        #how to know what type of variable?
        
        states = torch.tensor(states, dtype = torch.float)
        actions = torch.tensor(actions, dtype = torch.int64) # Ensure actions is [batch_size, 1]
        rewards = torch.tensor(rewards, dtype = torch.float)
        #print("pre-actions: ", actions, " shape: ", actions.shape)
        #print("rewards: ", rewards, " shape: ", rewards.shape)
        #Compute action probabilities (I assume this includes all iterations of states so will give a lot of sets of probabilities of actions)
        
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            #entering as just a number so need to do twice
            actions = torch.unsqueeze(actions,0)
            rewards = torch.unsqueeze(rewards, 0)
        action_probs = self.model(states)
        #print("actions: ", actions, " shape: ", actions.shape)
        #print("rewards: ", rewards, " shape: ", rewards.shape)
        #print("action_probs: ", action_probs, " shape: ", action_probs.shape)
        #print("states: ", states, " shape: ", states.shape)
        #Finds probability of the selected actions (in actions)
        #dim = 1 means it will get it from action in (batchsize, action). Dim = 0 would be batch# in (batchsize,action)
        #squeeze i.e. [138.7333] -> 138.733
        log_selected_action_probs = torch.log(action_probs.gather(1, actions).squeeze())

        #assuming since it knows the shape, it'll calculate based on the shape
        #Average of the losses for each actions for the batch
        loss = -torch.mean(log_selected_action_probs * rewards)

        # Zero (clears) gradients, perform a backward pass, and update the weights
        self.optimizer.zero_grad() #backprop saves all gradients so clear it all for this batch
        loss.backward()
        self.optimizer.step() #updates based on gradients from backwards()
        #total loss of predicted to target 
        return loss.item()
       
       
        '''
    Want to see if i can make it look like this instead of needing to separate out to two different code for batch and non-batch like above
     if len(action_probs.shape) == 3:
            losses = []
            for i in range(action_probs.size(0)):
                action_probs_sample = action_probs[i]
                #compare the action at ith position on dim = 1
                selected_action_probs = torch.gather(action_probs_sample, 1, actions[i])
                loss = -torch.log(selected_action_probs) * rewards[i]  # Calculate loss for the i-th sample with log
                losses.append(loss)  # Append the loss to the list of losses
            # Calculate the average loss for the batch
            avg_loss = sum(losses) / len(losses)
            # Zero (clears) gradients, perform a backward pass, and update the weights
            self.optimizer.zero_grad() #backprop saves all gradients so clear it all for this batch
            avg_loss.backward()
            self.optimizer.step() #updates based on gradients from backwards()
            #total loss of predicted to target 
            return avg_loss.item()

        else:
            selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            #assuming since it knows the shape, it'll calculate based on the shape
            #Average of the losses for each actions for the batch
            loss = -torch.log(selected_action_probs) * rewards

            # Zero (clears) gradients, perform a backward pass, and update the weights
            self.optimizer.zero_grad() #backprop saves all gradients so clear it all for this batch
            loss.backward()
            self.optimizer.step() #updates based on gradients from backwards()
            #total loss of predicted to target 
            return loss.item()
'''
        