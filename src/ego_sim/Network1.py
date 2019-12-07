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
from random_path_generator import RandomPathGenerator
from ego_sim import EgoSim
from stanley_pid import StanleyPID
from nn2_control import NN2Control
import matplotlib.pyplot as plt
from evolutionary_algorithm import EvolutionaryAlgorithm
from Min_dist_test import calc_off_tracking
import test_suite


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(6, 10)  # 6*6 from image dimension
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))*4/5*np.pi-2*np.pi/5

        return x

def test_network(network,x_true,y_true,vel,t):
    # Initialize truck simulator and control objects
    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    controller = NN2Control()
    pid=StanleyPID()
    
    x = []
    th1t=0
    th2t=0
    #do some random stuff
    ego2=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    
    for i in range(0,len(t)):
        
        network=network.float()
        state = ego.convert_world_state_to_front()
        state1 = ego2.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t,network)
        ctrl_delta_pid, ctrl_vel_pid, err_pid, interr_pid, differr_pid = pid.calc_steer_control(t[i],state1,x_true,y_true, vel)
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x1,y1,delt,tha,thb=ego2.simulate_timestep([ctrl_vel_pid,ctrl_delta_pid])
        x.append(err[0]*err[0])
        
    x=sum(x)/len(t)
    return  x
     
def train_network(network,k_crosstrack,k_heading): 
    rpg = RandomPathGenerator()
    x_true, y_true, t, vel = rpg.get_harder_path()
    xt1=x_true
    yt1=y_true
    t12=t
    vel1=vel
    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    controller = NN2Control()
    pid=StanleyPID(k_crosstrack=k_crosstrack,k_heading=k_heading)
    pid2=StanleyPID(k_crosstrack=k_crosstrack,k_heading=k_heading)
    learning_rate=0.01
    x = []
    y = []
    xp=[]
    yp=[]
    delta = []
    th1 = []
    th2 = []
    th1t=0
    th2t=0
    running_loss=0
    #do some  random stuff
    ego2=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    
    for i in range(0,len(t)):
        
        network=network.float()
        state = ego.convert_world_state_to_front()
        state1 = ego2.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t,network)
        ctrl_delta_pid, ctrl_vel_pid, err_pid, interr_pid, differr_pid = pid.calc_steer_control(t[i],state1,x_true,y_true, vel)

        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x1,y1,delt,tha,thb=ego2.simulate_timestep([ctrl_vel_pid,ctrl_delta_pid])
        xp.append(x1); yp.append(y1)
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)

        inputs=np.concatenate((err,ctrl_vel,interr[0],differr[0]),axis=None)
        inputs=np.append(inputs,th1t-th2t)

        network_input=torch.tensor(inputs)

        network= network.double()

        out=network(network_input)

    print(running_loss)
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.plot(xp,yp,'g--')
    plt.legend(['Network Performance','True Path', 'PID Performance'])
    plt.xlabel('X location, (m)')
    plt.ylabel('Y Location, (m)')
    plt.show()

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
    th1t=0
    th2t=0
    for i in range(200000):
        pid_list=pid.control_from_random_error()
        input1=pid_list[0:3]
        input2=pid_list[4:]
        input1=np.concatenate((input1,input2), axis=0)
        input1= np.append(input1,np.pi*np.random.random()-np.pi/2)
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
        if (i%200==199) :
            print(i)
            loss_time.append(i)
            plot_loss.append(pl/200)
            pl=0

        running_loss += loss.item()

        for f in network.parameters():
            f.data.sub_(f.grad.data * learning_rate)
    print(running_loss)
    plt.plot(loss_time,plot_loss)
    plt.xlabel('Iteration Number')
    plt.ylabel('Average Loss of Past 200 Iterations, (rad)')
    plt.show()
    #follow the path
    running_loss=0

    ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    ego2=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    controller = NN2Control()
    pid=StanleyPID(k_crosstrack=k_crosstrack,k_heading=k_heading)
    pid2=StanleyPID(k_crosstrack=k_crosstrack,k_heading=k_heading)
    pl=0
    plot_loss=[]
    for i in range(0,len(t)):
        network=network.float()
        state = ego.convert_world_state_to_front()
        state1 = ego2.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, network)
        ctrl_delta_pid, ctrl_vel_pid, err_pid, interr_pid, differr_pid = pid2.calc_steer_control(t[i],state,x_true,y_true, vel)
        ctrl_delta_pid1, ctrl_vel_pid1, err_pid1, interr_pid1, differr_pid1 = pid.calc_steer_control(t[i],state1,x_true,y_true, vel)
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x1,y1,delt,tha,thb=ego2.simulate_timestep([ctrl_vel_pid1,ctrl_delta_pid1])
        xp.append(x1); yp.append(y1)
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        inputs=np.concatenate((err,ctrl_vel,interr[0],differr[0]),axis=None)
        inputs=np.append(inputs, th1t-th2t)
        network_input=torch.tensor(inputs)
        network_target=torch.tensor(ctrl_delta_pid)
        network_target=network_target.double()
        network= network.double()
        out=network(network_input)
        network.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(out, network_target)
        loss.backward()
        if (i%len(t)/20)==(len(t)/20-1):
            plot_loss.append(pl/len(t)*20)
            loss_time.append(i)
            pl=0
        running_loss += loss.item()
        for f in network.parameters():
            f.data.sub_(f.grad.data * learning_rate)
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.plot(xp,yp,'g--')
    plt.legend(['Network Performance','True Path', 'PID Performance'])
    plt.xlabel('X location, (m)')
    plt.ylabel('Y Location, (m)')
    plt.show()
    
def train_nn_from_pid(k_crosstrack = {'P':20, 'I':2, 'D':5}, 
              k_heading = {'P':-0.5, 'I':0, 'D':0}):
    network=Net2()
    network.zero_grad()
    #Train the network until it is sufficient, asking the human operator for input on whether the point it reached  is  good enough
    network=network.float()
    for i in range(10):
        train_network(network,k_crosstrack,k_heading)
        a=input('is this good enough?')
        if a=='1':
            break
        
    return network

def test_network_on_benchmark(network,Benchmark):
    #Initialize varriables to run the first benchmark test  on the PID mimicking controller
    network=network.float()

    controller = NN2Control()
    x_true=Benchmark[:,0]
    y_true= Benchmark[:,1]
    t= Benchmark[:,2]
    vel=Benchmark[:,3]
    xp=[]
    yp=[]
    x=[]
    y=[]
    pid=StanleyPID()
    
    #Run the same benchmark on both the PID controller and the PID mimicking network and compare  the two
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
                ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, network)
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                x.append(xt); y.append(yt);
        if i == 1:
            pid_fitness, CTerr =calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
        else:
            controller_fitness, CTerr = calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
    print('Benchmark PID fitness: ', pid_fitness)
    print('Benchmark controller fitness: ', controller_fitness)
    plt.plot(x,y)
    plt.plot(x_true,y_true,'r--')
    plt.plot(xp,yp,'g--')
    plt.legend(['Network Performance','True Path', 'PID Performance'])
    plt.show()

def main():
#    network=Net2()
#    network.zero_grad()
    
    #Read in the benchmark paths that we will use
    Benchmark1=pd.read_csv('Benchmark_DLC_31ms_reduced.csv',sep=',',header=0)
    Benchmark1=Benchmark1.values
    Benchmark2=pd.read_csv('Benchmark_SScorner_80m_left_reduced.csv',sep=',',header=0)
    Benchmark2=Benchmark2.values
    Benchmark3=pd.read_csv('Benchmark_SScorner_500m_left_25ms_reduced.csv',sep=',',header=0)
    Benchmark3=Benchmark3.values


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

    
    #Train the network until it is sufficient, asking the human operator for input on whether the point it reached  is  good enough
    network=network.float()
    for i in range(10):
        train_network(network)
        a=input('is this good enough?')
        if a=='1':
            break

    #Initialize varriables to run the first benchmark test  on the PID mimicking controller
    network=network.float()

    controller = NN2Control()
    x_true=Benchmark1[:,0]
    y_true= Benchmark1[:,1]
    t= Benchmark1[:,2]
    vel=Benchmark1[:,3]
    xp=[]
    yp=[]
    x=[]
    y=[]
    pid=StanleyPID()
    
    #Run the same benchmark on both the PID controller and the PID mimicking network and compare  the two
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
                ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, network)
                xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                x_truck.append(xt)
                y_truck.append(yt)
                th1.append(th1t)
                th2.append(th2t)
                x.append(xt); y.append(yt);
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
    plt.xlabel('X location, (m)')
    plt.ylabel('Y Location, (m)')
    plt.show()

    #send the pid mimicking controller to the  evolutionary algorithm
    print('bias   value before  evo: ', network.fc3.bias.data)
    evolution=EvolutionaryAlgorithm(network)
    for i in range(1000):
        print(i)
        evolution.iterate()
        #every 20 steps, run a benchmark on the best controller in the system to see how it is progressing
        if i%20==0:
            Fc1=network.fc1.weight.data.numpy()
            Fc2=network.fc2.weight.data.numpy()
            Fc3=network.fc3.weight.data.numpy()
            Evo1=evolution.controllers[evolution.best_controller_idx].fc1.weight.data.numpy()
            Evo2=evolution.controllers[evolution.best_controller_idx].fc2.weight.data.numpy()
            Evo3=evolution.controllers[evolution.best_controller_idx].fc3.weight.data.numpy()
            print((Fc1-Evo1))
            print(np.linalg.norm((Fc1-Evo1)))
            print((Fc2-Evo2))
            print(np.linalg.norm((Fc2-Evo2)))
            print((Fc3-Evo3))
            print(np.linalg.norm((Fc3-Evo3)))
            Fc1b=network.fc1.bias.data.numpy()
            Fc2b=network.fc2.bias.data.numpy()
            Fc3b=network.fc3.bias.data.numpy()
            Evo1b=evolution.controllers[evolution.best_controller_idx].fc1.bias.data.numpy()
            Evo2b=evolution.controllers[evolution.best_controller_idx].fc2.bias.data.numpy()
            Evo3b=evolution.controllers[evolution.best_controller_idx].fc3.bias.data.numpy()
            print((Fc1b-Evo1b))
            print(np.linalg.norm((Fc1b-Evo1b)))
            print((Fc2b-Evo2b))
            print(np.linalg.norm((Fc2b-Evo2b)))
            print((Fc3b-Evo3b))
            print(np.linalg.norm((Fc3b-Evo3b)))
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
            plt.xlabel('X location, (m)')
            plt.ylabel('Y Location, (m)')
            plt.show()
    
    #Initialize varriables to run the first benchmark test
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
    #after the network has finished its evolutionary training, check it  the first benchmark
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
    plt.xlabel('X location, (m)')
    plt.ylabel('Y Location, (m)')
    plt.show()
    
    #Initialize varriables to run the second benchmark test on the controller trained on the
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
    #check it  the second benchmark
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
    plt.xlabel('X location, (m)')
    plt.ylabel('Y Location, (m)')
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
    plt.xlabel('X location, (m)')
    plt.ylabel('Y Location, (m)')
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
if __name__ == "__main__":
    main()