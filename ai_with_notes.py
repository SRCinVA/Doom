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
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=5) # strange how "self." is immediately in the variable name.
                                                                                # Conv2d is built into nn
                                                                                # you need 3 in-channels when dealing with colors; just 1 for B&W.
                                                                                # out-channels is equal to the number of features you want to detect in your original images. 
                                                                                # 32 is the norm; from the original, it will produce 32 images with detected features
                                                                                # kernel size = the dimensions of the square that will go through the original image.
                                                                                # for the first one, we'll use 5*5 feature detector, which will decrease through the next convolutional layers.
        # this builds on layer 1 and reinforces it
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) # we'll stick with 32 new images, like before. Then we'll reduce the kernel size.
        
        # this generates even more details from the previous layers. We'll switch to 64 from 32. This is a classic pattern.
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        # realize that this is a mind-blowing number of images to deal with from these convolutions: 32*32*64.
        # next, this is how we flatten the information into a vector for feeding into the NN as input.
        # we'll have one hidden layer and an output layer
        self.fc1 # input to hidden
        self.fc2 # hidden to output


# Building the body

# Part 2- Implementing Deep Convolutional Q-Learning
