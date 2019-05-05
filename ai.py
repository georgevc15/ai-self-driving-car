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
    
    #capacity: transitions or steps to be remembered
    def __init__(self, capacity):
        self.cpacity = capacity
        self.memory = []