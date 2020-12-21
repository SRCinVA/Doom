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



# Part 2- Implementing Deep Convolutional Q-Learning