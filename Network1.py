#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:35:40 2019

@author: orochi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        #self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7, 10)  # 6*6 from image dimension
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #print(np.shape(x))
        # If the size is a square you can only specify a single number
        #a=F.relu(self.conv2(x))
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #print('a: ', a)
        #print('x: ',x)
       # x = x.view(-1, self.num_flat_features(x))
        #print(np.shape(x))
        x = F.relu(self.fc1(x))
        #print(np.shape(x))
        x = F.relu(self.fc2(x))
        #print(np.shape(x))
        x = self.fc3(x)
        #print(np.shape(x))
        return x
    
def main():
    network=Net()
    network.zero_grad()
    learning_rate = 0.01
    for i in range(1000):
        out=network(input1)
        network.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(out, target1)
        loss.backward()
        for f in network.parameters():
            f.data.sub_(f.grad.data * learning_rate)

def calculateError():
    
    