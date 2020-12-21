#AI for Doom

# Import the libraries 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#import the packags for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# import other Python files
import experience_replay, image_preprocessing

# Part 1 - Building the AI

# Building the brain

class CNN(nn.Module):  # the CNN inherits from the nn module

    def __init__(self, number_actions): # we'll be importing the number of actions from the Doom environment
                                        # it's imported from the Doom environment wrappers, enabling us to test this AI 
                                        # in other Doom environments.
        super(CNN, self).__init__()     # the super function activates inheritance. Question: the .__init__ usage looks very strange.
        # this AI will have "eyes," which will be the convolutional layer of the CNN.
        # from the convolutional layers, the information is passed to a classic NN (input, hidden, and output layers) 
        # the next stage of processing. That's where it predicts the Q values of each possible move.
        
        # now we'll define 5 variables: 3 for convolutional layers, and 2 for the hidden layers             
        # these layers are the AI's eyes into the environment and how it will govern its behavior while in the game.
        self.convolution1 # strange how "self." is immediately in the variable name.
        self.convolution2 # this builds on layer 1 and reinforces it
        self.convolution3 # this generates even more details from the previous layers
        # next, this is how we flatten the information into a vector for feeding into the NN as input.
        # we'll have one hidden layer and an output layer
        self.fc1 # input to hidden
        self.fc2 # hidden to output


# Building the body

# Part 2- Implementing Deep Convolutional Q-Learning
