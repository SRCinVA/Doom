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
        
        # input from the gigantic vector to the hidden layer:
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,80,80)), out_features = 40)
        # in_features are the total number of pixels in the massive vector.
        # we need to make a function to find that number; let's just drop a placeholder in there for the moment.
        # outfeatures are the number of neurons in the hidden layer; we can select this.

        # hidden to output layer (reflective of the Q values associated with actions):
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)
        # Logically, it's 40 from fc1 (that's easy).
        # Each output neuron corresponds to one Q value.
        # Each Q value corresponds to one possible action.
        # we have passed this exact figure in as number_actions into the function.
        
    def count_neurons(self, image_dim): # image_dim are the dimensions of the images coming from Doom (80*80)
        x = Variable(torch.rand(1, *image_dim)) # we create a "fake" image here first; hard to understand his reasoning here: the batch (1) and (80*80) for the dimensions of the input image.
                                    # this is just to help us get the final number of neurons, whether or not they would be used in Doom.
                                    # with Variable, it is an input image of random pixels that was just converted into a torch variable
                                    # that will go into the convolutional layers of the neural network.
                                    # the number of neurons we want is between convol3 and fc1.
        # now we have to propogate the image into the NN to reach the flattening layer.
        # that propogation will require 3 steps.
        # 1.) apply the convolution to the input images; 
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))  # we'll choose 3 for the kernel size, with a stride of 2 pixels.
        # 2.) (above) apply max pooling to the convoluted images;
        # 3.) (also above) we'll apply Relu to activate the neurons for these pooled convoluted images.
        
        # now let's apply convolution 2 to "x"; then we'll mx pool it and activate its neurons.
        # we'll choose 3 for the kernel size, with a stride of 2 pixels.
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))

        # now on to the third convolutional layer
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        # now on to the flattening layer: we'll take all the pixels of the 3rd convolutonal layer
        # we'll put them into a huge vector (the flattening layer)
        
        # we're looking for the number of neurons in the flattening layer
        return x.data.view(1, -1).size(1)    
        # with size, we line up all the pixels one by one, and this will be the input of the fully connected network. 
        # with data.view, we can see how many neurons there are.
        
        # with these two functions, we have (1.) architected the network and (2.) counted the neurons


# Building the body

# Part 2- Implementing Deep Convolutional Q-Learning
