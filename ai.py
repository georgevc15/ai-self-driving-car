# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Creating the architecture of the Nurall Network

class Network(nn.Module):
    
    #input_size -> Vectors:  3 signals and orientation / -orientation
    #nb_action ->  Actions: left, right and straight
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    #x is the hidden neurons
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
 
    
#Implementing Experience Replay
    
class ReplayMemory(object):
    
    #capacity: events or steps to be remembered
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
# Implementing Deep Q Learning  
        
    class Dqn():
        
        #gamma delay operator
        def __init__(self, input_size, nb_action, gamma):
            self.gamma = gamma
            self.reward_window = []
            #create one neural network for the Deep Q Learning model
            self.model = Network(input_size, nb_action, gamma)
            #create memory 
            self.memory = ReplayMemory(100000)
            self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
            self.last_state = torch.Tensor(input_size).unsqueeze(0)
            self.last_action = 0
            self.last_reward = 0