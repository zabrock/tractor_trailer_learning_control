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
from random_path_generator import RandomPathGenerator
from ego_sim import EgoSim
from stanley_pid import StanleyPID
from nn_control import NNControl
from nn2_control import NN2Control
import matplotlib.pyplot as plt
import csv
import time
import math
from evolutionary_algorithm import EvolutionaryAlgorithm
from Min_dist_test import calc_off_tracking

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(7, 10)  
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
        x = torch.sigmoid(self.fc1(x))
        #print(np.shape(x))
        x = torch.sigmoid(self.fc2(x))
        #print(np.shape(x))
        x = torch.sigmoid(self.fc3(x))*4/5*np.pi-2*np.pi/5
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
        self.fc1 = nn.Linear(8, 10)  # 6*6 from image dimension
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
        x = torch.sigmoid(self.fc1(x))
        #print(np.shape(x))
        x = torch.sigmoid(self.fc2(x))
        #print(np.shape(x))
        x = torch.sigmoid(self.fc3(x))*4/5*np.pi-2*np.pi/5
        #print(np.shape(x))
        return x

def test_network(network,x_true,y_true,vel,t):
    rpg = RandomPathGenerator()
    #x_true, y_true, t, vel = rpg.get_random_path()
    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    controller = NNControl()
    pid=StanleyPID()
    pid2=StanleyPID()
    learning_rate=0.01
    x = []
    y = []
    xp=[]
    yp=[]
    delta = []
    th1 = []
    th2 = []
    running_loss=0
    #do some  random stuff
    ego2=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    
    for i in range(0,len(t)):
        
        network=network.float()
        state = ego.convert_world_state_to_front()
        state1 = ego2.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, network)
        ctrl_delta_pid, ctrl_vel_pid, err_pid, interr_pid, differr_pid = pid.calc_steer_control(t[i],state1,x_true,y_true, vel)
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
        x1,y1,delt,tha,thb=ego2.simulate_timestep([ctrl_vel_pid,ctrl_delta_pid])
        #xp.append(x1); yp.append(y1)
        #x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        x.append(err[0]*err[0])
                #print(np.shape(input1))
        #print(network.parameters())
        #print(network.fc1.weight.data)
        #print(network.fc3.weight.data)
        #print(j)
        #print(err)
        #print(ctrl_vel)
        #print(interr)
        #print(differr)
        #inputs=np.concatenate((err,ctrl_vel,interr,differr),axis=None)
        #inputs=[err[],err[],ctrl_vel,interr[],interr[],differr]
        #network_input=torch.tensor(inputs)
        #print(network_input)
        #network_target=torch.tensor(ctrl_delta_pid)
        #network_target=network_target.double()
        #network= network.double()
        #print(network_input,network_target)
        #out=network(network_input)
    x=sum(x)/len(t)
    return  x
 
    
def train_network(network): 
    rpg = RandomPathGenerator()
    x_true, y_true, t, vel = rpg.get_random_path()
    '''while True:
        plt.plot(x_true,y_true)
        plt.show()
        a=input('good?')
        if a=='1':
            break
        else:
            x_true, y_true, t, vel = rpg.get_random_path()
    '''
    xt1=x_true
    yt1=y_true
    t12=t
    vel1=vel
    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    controller = NNControl()
    pid=StanleyPID()
    pid2=StanleyPID()
    learning_rate=0.01
    x = []
    y = []
    xp=[]
    yp=[]
    delta = []
    th1 = []
    th2 = []
    running_loss=0
    #do some  random stuff
    ego2=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    
    for i in range(0,len(t)):
        
        network=network.float()
        state = ego.convert_world_state_to_front()
        state1 = ego2.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, network)
        ctrl_delta_pid, ctrl_vel_pid, err_pid, interr_pid, differr_pid = pid.calc_steer_control(t[i],state1,x_true,y_true, vel)
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
        x1,y1,delt,tha,thb=ego2.simulate_timestep([ctrl_vel_pid,ctrl_delta_pid])
        xp.append(x1); yp.append(y1)
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
        #network_target=torch.tensor(ctrl_delta_pid)
        #network_target=network_target.double()
        network= network.double()
        #print(network_input,network_target)
        out=network(network_input)
        #if (i >50) &  (i < 100):
        #print(out)
        #network.zero_grad()
        #criterion = nn.MSELoss()
        #loss = criterion(out, network_target)
        #loss.backward()
        #running_loss += loss.item()
        #print(loss.item())
        #for f in network.parameters():
        #    f.data.sub_(f.grad.data * learning_rate)
    print(running_loss)
    #plt.plot(x,y)
    #plt.plot(x_true,y_true,'r--')
    #plt.plot(xp,yp,'g--')
    #plt.legend(['Network Performance','True Path', 'PID Performance'])
    #plt.show()
    running_loss=0
    x = []
    y = []
    xp=[]
    yp=[]
    delta = []
    th1 = []
    th2 = []
    pl=0
    plot_loss=[]
    loss_time=[]
    MSE=[]
    MSE1=0
    for i in range(20000):
        #network=network.float()
        #state = ego.convert_world_state_to_front() 
        #ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, network)
        pid_list=pid.control_from_random_error()
        input1=pid_list[0:3]
        input2=pid_list[4:]
        input1=np.concatenate((input1,input2), axis=0)
        #print(input1)
        target1=pid_list[3]
        network_input=torch.tensor(input1)
        #print(network_input)
        network_target=torch.tensor(target1)
        network_target=network_target.double()
        network= network.double()
        #print(network_input)
        out=network(network_input)
        network.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(out, network_target)
        loss.backward()
        pl+=loss.item()
        if (i%200==199)& (i<2000) :
            print(i)
            MSE1=test_network(network,xt1,yt1,vel1,t12)
            MSE.append(MSE1)
            loss_time.append(i)
        running_loss += loss.item()
        #if  i % 10000==0:
        #    print(i)
        #print(out.data,network_target.data, out.data-network_target.data)
        #print(loss.item())
        for f in network.parameters():
            f.data.sub_(f.grad.data * learning_rate)
    print(running_loss)
    #follow the path
    running_loss=0
    #plt.plot(loss_time,MSE)
    #plt.xlabel('Data Number')
    #plt.ylabel('Average Fitness')
    #plt.show()

    #x_true, y_true, t, vel = rpg.get_random_path()
    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    ego2=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    controller = NNControl()
    pid=StanleyPID()
    pid2=StanleyPID()
    pl=0
    plot_loss=[]
    for i in range(0,len(t)):
        
        network=network.float()
        state = ego.convert_world_state_to_front()
        state1 = ego2.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, network)
        ctrl_delta_pid, ctrl_vel_pid, err_pid, interr_pid, differr_pid = pid2.calc_steer_control(t[i],state,x_true,y_true, vel)
        ctrl_delta_pid1, ctrl_vel_pid1, err_pid1, interr_pid1, differr_pid1 = pid.calc_steer_control(t[i],state1,x_true,y_true, vel)
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
        x1,y1,delt,tha,thb=ego2.simulate_timestep([ctrl_vel_pid1,ctrl_delta_pid1])
        xp.append(x1); yp.append(y1)
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
        #if (i >50) &  (i < 100):
        #print(out)
        network.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(out, network_target)
        loss.backward()
        if (i%len(t)/20)==(len(t)/20-1):
            plot_loss.append(pl/len(t)*20)
            loss_time.append(i)
            pl=0
            
            
        running_loss += loss.item()
        #print(loss.item())
        for f in network.parameters():
            f.data.sub_(f.grad.data * learning_rate)
    '''
    print(running_loss)
    plt.plot(loss_time,plot_loss)
    plt.xlabel('Data Number')
    plt.ylabel('Average Loss of 500 samples')
    plt.show()
    '''
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.plot(xp,yp,'g--')
    plt.legend(['Network Performance','True Path', 'PID Performance'])
    plt.show()
    

def main():
    network=Net1()
    #print(network.fc1.weight.detach().numpy())
    network.zero_grad()
    network2=Net2()
    Benchmark1=pd.read_csv('Benchmark_DLC_31ms_reduced.csv',sep=',',header=0)
    Benchmark1=Benchmark1.values
    Benchmark2=pd.read_csv('Benchmark_SScorner_80m_left_reduced.csv',sep=',',header=0)
    Benchmark2=Benchmark2.values
    Benchmark3=pd.read_csv('Benchmark_SScorner_500m_left_25ms_reduced.csv',sep=',',header=0)
    Benchmark3=Benchmark3.values
    #PID_Data=np.random.rand(100,5)
    #print('PID',PID_Data)
    #PID_Data=pd.read_csv('random_path_pid_more_output_0.csv',sep=',',header=0)
    #print(PID_Data)
    #PID_Data=PID_Data.values
    #np.random.shuffle(PID_Data)
    #input1=PID_Data[:,0:3]
    #input2=PID_Data[:,4:]
    #input1=np.concatenate((input1,input2), axis=1)
    #print(input1)
    #target1=PID_Data[:,3]
    '''$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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
            #if running_loss >= 5:
                #input('press  enter')
            total_loss+=running_loss
            PID_Data=pd.read_csv('random_path_pid_more_output_'+str(i)+'.csv',sep=',',header=0)
        #print(PID_Data)
            
            PID_Data=PID_Data.values
            np.random.shuffle(PID_Data)
            input1=PID_Data[:,0:2]
            input2=PID_Data[:,4:]
            input1=np.concat enate((input1,input2),axis=1)
        #print(input1)
            target1=PID_Data[:,3]
        print('total loss this set: ', total_loss)
        #print('[%5d] loss: %.3f' %
        #(i + 1, running_loss))
    '''   

    #running_loss = 0.0
    
    
    
    network=network.float()
    for i in range(10):
        train_network(network)
        a=input('is this good enough?')
        if a=='1':
            break
        
    '''
    for i  in range(10):
        train_network(network)
        j=input('is this good enough? 1 for yes  0 for no')
        #print(j)
        if j =='1':
            break
    b=time.process_time()
    '''
    #print(a-b)
    
    
    
    '''
    with open('weights.csv', mode='w') as weights:
        weight_writer = csv.writer(weights, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        weight_writer.writerow(network.fc1.weight.detach().numpy())
        weight_writer.writerow(network.fc1.bias.detach().numpy())
        weight_writer.writerow(network.fc2.weight.detach().numpy())
        weight_writer.writerow(network.fc2.bias.detach().numpy())
        weight_writer.writerow(network.fc3.weight.detach().numpy())
        weight_writer.writerow(network.fc3.bias.detach().numpy())
    '''
    #test_network(network)
    network=network.double()
    network2.fc1.weight.data[:,0:7]=network.fc1.weight.data
    network2.fc2.weight.data=network.fc2.weight.data
    network2.fc3.weight.data=network.fc3.weight.data
    network2.fc1.bias.data=network.fc1.bias.data
    network2.fc2.bias.data=network.fc2.bias.data
    network2.fc3.bias.data=network.fc3.bias.data
    network2.fc1.weight.data[:,7]=torch.zeros(10)
    network2=network2.float()
    controller = NN2Control()
    x_true=Benchmark1[:,0]
    y_true= Benchmark1[:,1]
    t= Benchmark1[:,2]
    vel=Benchmark1[:,3]
    x=[]
    y=[]
    xp=[]
    yp=[]

    x_truck=[]
    y_truck=[]
    th1t=0
    th2t=0
    th1=[]
    th2=[]
    pid=StanleyPID()
    for i in  range(2):
        ego=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        print('controller: ', i)
        th1t=0
        th2t=0
        th1=[]
        th2=[]
        x_truck=[]
        y_truck=[]
        for j in range(len(t)):
            if i == 1:
                state = ego.convert_world_state_to_front()
                ctrl_delta, ctrl_vel, err, interr, differr = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                xp.append(xt); yp.append(yt)
            else:
                state = ego.convert_world_state_to_front()
                ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, network2)
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                x.append(xt); y.append(yt);
            #inputs=np.concatenate((err,ctrl_vel,interr,differr),axis=None)
            #network_input=torch.tensor(inputs)
            #out=self.controllers[i](network_input)
            #x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        if i == 1:
            pid_fitness, CTerr =calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
        else:
            controller_fitness, CTerr = calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
    print('Benchmark 1 PID fitness: ', pid_fitness)
    print('Benchmark 1 controller fitness: ', controller_fitness)
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.plot(xp,yp,'g--')
    plt.legend(['Network Performance','True Path', 'PID Performance'])
    plt.show()
    #print(network.fc1.weight.data)
    #print(network.fc3.weight.data)
    #print(network2.fc1.weight.data)
    #print(network2.fc3.weight.data)
    network2=network2.double()
    evolution=EvolutionaryAlgorithm(network2)
    for i in range(1000):
        print(i)
        evolution.iterate()
        if i%20==0:
            controller = NN2Control()
            x_true=Benchmark1[:,0]
            y_true= Benchmark1[:,1]
            t= Benchmark1[:,2]
            vel=Benchmark1[:,3]
            x=[]
            y=[]
            xp=[]
            yp=[]
        
            x_truck=[]
            y_truck=[]
            th1t=0
            th2t=0
            th1=[]
            th2=[]
            pid=StanleyPID()
            for i in  range(2):
                ego=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
                print('controller: ', i)
                th1t=0
                th2t=0
                th1=[]
                th2=[]
                x_truck=[]
                y_truck=[]
                for j in range(len(t)):
                    if i == 1:
                        state = ego.convert_world_state_to_front()
                        ctrl_delta, ctrl_vel, err, interr, differr = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
                        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                        x_truck.append(xt)
                        y_truck.append(yt)
                        th1.append(th1t)
                        th2.append(th2t)
                        xp.append(xt); yp.append(yt)
                    else:
                        state = ego.convert_world_state_to_front()
                        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, evolution.controllers[evolution.best_controller_idx])
                        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                        x_truck.append(xt)
                        y_truck.append(yt)
                        th1.append(th1t)
                        th2.append(th2t)
                        x.append(xt); y.append(yt);
                    #inputs=np.concatenate((err,ctrl_vel,interr,differr),axis=None)
                    #network_input=torch.tensor(inputs)
                    #out=self.controllers[i](network_input)
                    #x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
                if i == 1:
                    pid_fitness, CTerr =calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
                else:
                    controller_fitness, CTerr = calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
            
            print('Benchmark 1 PID fitness: ', pid_fitness)
            print('Benchmark 1 controller fitness: ', controller_fitness)
            plt.plot(x,y)
            plt.plot(x_true,y_true,'r--')
            plt.plot(xp,yp,'g--')
            plt.legend(['Network Performance','True Path', 'PID Performance'])
            plt.show()
    
    controller = NN2Control()
    x_true=Benchmark1[:,0]
    y_true= Benchmark1[:,1]
    t= Benchmark1[:,2]
    vel=Benchmark1[:,3]
    x=[]
    y=[]
    xp=[]
    yp=[]

    x_truck=[]
    y_truck=[]
    th1t=0
    th2t=0
    th1=[]
    th2=[]
    pid=StanleyPID()
    for i in  range(2):
        ego=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        print('controller: ', i)
        th1t=0
        th2t=0
        th1=[]
        th2=[]
        x_truck=[]
        y_truck=[]
        for j in range(len(t)):
            if i == 1:
                state = ego.convert_world_state_to_front()
                ctrl_delta, ctrl_vel, err, interr, differr = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                xp.append(xt); yp.append(yt)
            else:
                state = ego.convert_world_state_to_front()
                ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, evolution.controllers[evolution.best_controller_idx])
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                x.append(xt); y.append(yt);
            #inputs=np.concatenate((err,ctrl_vel,interr,differr),axis=None)
            #network_input=torch.tensor(inputs)
            #out=self.controllers[i](network_input)
            #x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        if i == 1:
            pid_fitness, CTerr =calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
        else:
            controller_fitness, CTerr = calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
    
    print('Benchmark 1 PID fitness: ', pid_fitness)
    print('Benchmark 1 controller fitness: ', controller_fitness)
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.plot(xp,yp,'g--')
    plt.legend(['Network Performance','True Path', 'PID Performance'])
    plt.show()
    x_true=Benchmark2[:,0]
    y_true= Benchmark2[:,1]
    t= Benchmark2[:,2]
    vel=Benchmark2[:,3]
    
    x_truck=[]
    y_truck=[]
    th1t=0
    th2t=0
    th1=[]
    th2=[]
    pid=StanleyPID()
    x=[]
    y=[]
    xp=[]
    yp=[]
    for i in  range(2):
        ego=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        print('controller: ', i)
        th1t=0
        th2t=0
        th1=[]
        th2=[]
        x_truck=[]
        y_truck=[]
        for j in range(len(t)):
            if i == 1:
                state = ego.convert_world_state_to_front()
                ctrl_delta, ctrl_vel, err, interr, differr = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                xp.append(xt); yp.append(yt)
            else:
                state = ego.convert_world_state_to_front()
                ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, evolution.controllers[evolution.best_controller_idx])
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                x.append(xt); y.append(yt);
            #inputs=np.concatenate((err,ctrl_vel,interr,differr),axis=None)
            #network_input=torch.tensor(inputs)
            #out=self.controllers[i](network_input)
            #x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        if i == 1:
            pid_fitness, CTerr =calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
        else:
            controller_fitness, CTerr = calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
    print('Benchmark 2 PID fitness: ', pid_fitness)
    print('Benchmark 2 controller fitness: ', controller_fitness)
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.plot(xp,yp,'g--')
    plt.legend(['Network Performance','True Path', 'PID Performance'])
    plt.show()
    x_true=Benchmark3[:,0]
    y_true= Benchmark3[:,1]
    t= Benchmark3[:,2]
    vel=Benchmark3[:,3]
    x=[]
    y=[]
    xp=[]
    yp=[]
    x_truck=[]
    y_truck=[]
    th1t=0
    th2t=0
    th1=[]
    th2=[]
    pid=StanleyPID()
    for i in  range(2):
        ego=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        print('controller: ', i)
        th1t=0
        th2t=0
        th1=[]
        th2=[]
        x_truck=[]
        y_truck=[]
        for j in range(len(t)):
            if i == 1:
                state = ego.convert_world_state_to_front()
                ctrl_delta, ctrl_vel, err, interr, differr = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                xp.append(xt); yp.append(yt)
                
            else:
                state = ego.convert_world_state_to_front()
                ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, evolution.controllers[evolution.best_controller_idx])
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                x.append(xt); y.append(yt);
               
            #inputs=np.concatenate((err,ctrl_vel,interr,differr),axis=None)
            #network_input=torch.tensor(inputs)
            #out=self.controllers[i](network_input)
            #x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        if i == 1:
            pid_fitness, CTerr =calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
        else:
            controller_fitness, CTerr = calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
    print('Benchmark 3 PID fitness: ', pid_fitness)
    print('Benchmark 3 controller fitness: ', controller_fitness)
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.plot(xp,yp,'g--')
    plt.legend(['Network Performance','True Path', 'PID Performance'])
    plt.show()
    # 
    #controller=NNControl()
   
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