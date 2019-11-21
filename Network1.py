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
from src.ego_sim.nn_control import NNControl
import matplotlib.pyplot as plt

class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        #self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7, 5)  # 6*6 from image dimension
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
        x = torch.sigmoid(self.fc1(x))  
        #print(np.shape(x))
        #x = F.relu(self.fc2(x))
        #print(np.shape(x))
        x = torch.sigmoid(self.fc3(x))*np.pi-np.pi/2
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
        self.fc1 = nn.Linear(11, 5)  # 6*6 from image dimension
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

def test_network(network):
    rpg = RandomPathGenerator()
    x_true, y_true, t, vel = rpg.get_random_path()
    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    controller = NNControl()
    pid=StanleyPID()

    x = []
    y = []
    delta = []
    th1 = []
    th2 = []
    
    for i in range(0,len(t)):
        state = ego.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, network)
        ctrl_delta_pid, ctrl_vel_pid, err_pid, interr_pid, differr_pid = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
        #print([ct,hd,vel])
        #network_inputs=ct
        #np.append(network_inputs,hd,vel)
        #np.append(network_inputs,vel)
        #network_inputs.append(hd)
        #network_inputs.append(vel)
        #ctrl_delta = network(torch.tensor(network_inputs))
        #print([ctrl_delta,ctrl_vel])
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
    
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.show()
 
    
def train_network(network): 
    rpg = RandomPathGenerator()
    x_true, y_true, t, vel = rpg.get_random_path()
    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    controller = NNControl()
    pid=StanleyPID()
    learning_rate=0.01
    x = []
    y = []
    delta = []
    th1 = []
    th2 = []
    running_loss=0
    for i in range(0,len(t)):
        
        network=network.float()
        state = ego.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, network)
        ctrl_delta_pid, ctrl_vel_pid, err_pid, interr_pid, differr_pid = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
        #print(err, interr, differr)
        #print('network', ctrl_delta, ctrl_vel, 'pid', ctrl_delta_pid, ctrl_vel_pid)
        #print([ct,hd,vel])
        #network_inputs=ct
        #np.append(network_inputs,hd,vel)
        #np.append(network_inputs,vel)
        #network_inputs.append(hd)
        #network_inputs.append(vel)
        #ctrl_delta = network(torch.tensor(network_inputs))
        #print([ctrl_delta,ctrl_vel])
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
                #print(np.shape(input1))
        #print(network.parameters())
        #print(network.fc1.weight.data)
        #print(network.fc3.weight.data)
        #print(j)
        #print(err)
        #print(ctrl_vel)
        #print(interr)
        #print(differr)
        inputs=np.concatenate((err,ctrl_vel,interr,differr),axis=None)
        #inputs=[err[],err[],ctrl_vel,interr[],interr[],differr]
        network_input=torch.tensor(inputs)
        #print(network_input)
        network_target=torch.tensor(ctrl_delta_pid)
        network_target=network_target.double()
        network= network.double()
        #print(network_input,network_target)
        out=network(network_input)
        network.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(out, network_target)
        loss.backward()
        running_loss += loss.item()
        #print(loss.item())
        for f in network.parameters():
            f.data.sub_(f.grad.data * learning_rate)
    print(running_loss)
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.show()

def main():
    network=Net1()
    network.zero_grad()
    learning_rate = 0.01
    network2=Net2()
    #PID_Data=np.random.rand(100,5)
    #print('PID',PID_Data)
    PID_Data=pd.read_csv('random_path_pid_more_output_0.csv',sep=',',header=0)
    #print(PID_Data)
    PID_Data=PID_Data.values
    input1=PID_Data[:,0:3]
    input2=PID_Data[:,4:]
    input1=np.concatenate((input1,input2), axis=1)
    #print(input1)
    target1=PID_Data[:,3]
    for  k in range(5):
        thing_range=list(range(50))
        np.random.shuffle(thing_range)
        total_loss=0
        for i in thing_range:
            running_loss=0
            for j in range(len(PID_Data)):
                #print(np.shape(input1))
                #print(network.parameters())
                #print(network.fc1.weight.data)
                #print(network.fc3.weight.data)
                #print(j)
                network_input=torch.tensor(input1[j])
                #print(network_input)
                network_target=torch.tensor(target1[j])
                network_target=network_target.double()
                network= network.double()
                
                out=network(network_input)
                network.zero_grad()
                criterion = nn.MSELoss()
                loss = criterion(out, network_target)
                loss.backward()
                running_loss += loss.item()
                #print(out.data,network_target.data, out.data-network_target.data)
                #print(loss.item())
                for f in network.parameters():
                    f.data.sub_(f.grad.data * learning_rate)
            print('[%5d] loss: %.3f' % (i + 1, running_loss))
            if running_loss >= 5:
                input('press  enter')
            total_loss+=running_loss
            PID_Data=pd.read_csv('random_path_pid_more_output_'+str(i)+'.csv',sep=',',header=0)
        #print(PID_Data)
            np.random.shuffle(PID_Data)
            PID_Data=PID_Data.values
            input1=PID_Data[:,0:3]
            input2=PID_Data[:,4:]
            input1=np.concatenate((input1,input2),axis=1)
        #print(input1)
            target1=PID_Data[:,3]
        print('total loss this set: ', total_loss)
        #print('[%5d] loss: %.3f' %
        #(i + 1, running_loss))
        
    running_loss = 0.0
    network=network.float()
    for i  in range(10):
        train_network(network)
    test_network(network)
    '''
    network2.fc1.weight.data[:,0:3]=network.fc1.weight.data
    network2.fc3.weight.data[0]=network.fc3.weight.data
    network2.fc1.bias.data=network.fc1.bias.data
    network2.fc3.bias.data[0]=network.fc3.bias.data
    print(network.fc1.weight.data)
    print(network.fc3.weight.data)
    print(network2.fc1.weight.data)
    print(network2.fc3.weight.data)
    #controller=NNControl()
   ''' 
    #controller.calc_steer_control(t,state,path_x,path_y,path_vel,network)
    #a=network.parameters()
    #print(a)

    '''
    for i in range(0,len(t)):
        state = ego.convert_world_state_to_front()
        ctrl_delta, ctrl_vel = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
def calculateError():
    
    '''
main()