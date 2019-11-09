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
import pandas as pd
import numpy as np
from src.ego_sim.random_path_generator import RandomPathGenerator
from src.ego_sim.ego_sim import EgoSim
from src.ego_sim.stanley_pid import StanleyPID

class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        #self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3, 5)  # 6*6 from image dimension
        #self.fc2 = nn.Linear(10, 5)
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
        #x = F.relu(self.fc2(x))
        #print(np.shape(x))
        x = self.fc3(x)
        #print(np.shape(x))
        return x
  
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        #self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7, 5)  # 6*6 from image dimension
        #self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)

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
        #x = F.relu(self.fc2(x))
        #print(np.shape(x))
        x = self.fc3(x)
        #print(np.shape(x))
        return x

def transfer_weights(network1,network2):
    weights_2_transfer=[]
    for f in network1.parameters():
        weights_2_transfer.append()

def test_network(network):
    rpg = RandomPathGenerator()
    x_true, y_true, t, vel = rpg.get_random_path()
    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    pid = StanleyPID()
    

    x = []
    y = []
    delta = []
    th1 = []
    th2 = []
    
    for i in range(0,len(t)):
        state = ego.convert_world_state_to_front()
        _, _, ct, hd = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
        print([ct,hd,vel])
        network_inputs=ct
        np.append(network_inputs,hd,vel)
        #np.append(network_inputs,vel)
        #network_inputs.append(hd)
        #network_inputs.append(vel)
        ctrl_delta = network(torch.tensor(network_inputs))
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
    
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.show()

def main():
    network=Net1()
    network.zero_grad()
    learning_rate = 0.01
    #PID_Data=np.random.rand(100,5)
    #print('PID',PID_Data)
    PID_Data=pd.read_csv('random_path_pid_0.csv',sep=',',header=0)
    #print(PID_Data)
    PID_Data=PID_Data.values
    input1=PID_Data[:,0:3]
    target1=PID_Data[:,3:]
    for i in range(10):
        running_loss=0
        for j in range(len(PID_Data)):
            #print(np.shape(input1))
            #print(network.parameters())
            network_input=torch.tensor(input1[j])
            #print(network_input)
            network_target=torch.tensor(target1[j])
            network= network.double()
            out=network(network_input)
            network.zero_grad()
            criterion = nn.MSELoss()
            loss = criterion(out, network_target)
            loss.backward()
            running_loss += loss.item()
            #print(loss.item())
            for f in network.parameters():
                f.data.sub_(f.grad.data * learning_rate)
        np.random.shuffle(PID_Data)
        print(PID_Data[0])
        input1=PID_Data[:,0:3]
        target1=PID_Data[:,3:]
        print('[%5d] loss: %.3f' %
        (i + 1, running_loss))
    running_loss = 0.0
    #a=network.parameters()
    #print(a)
    #test_network(network)
    '''
    for i in range(0,len(t)):
        state = ego.convert_world_state_to_front()
        ctrl_delta, ctrl_vel = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
def calculateError():
    
    '''
main()