#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:34:22 2019

@author: Zeke
"""
from random_path_generator import RandomPathGenerator
from ego_sim import EgoSim
from stanley_pid import StanleyPID
from nn_control import NNControl
import numpy as np
import matplotlib.pyplot as plt
from Min_dist_test import calc_off_tracking
import pandas as pd

def basic_fitness_comparison(network,num_tests=10):
    # Initialize memory for fitness evaluation
    pid_fitness = np.zeros(num_tests)
    nn_fitness = 200*np.random.random(num_tests)
    rpg = RandomPathGenerator()
    for i in range(0,num_tests):
        print('{} of {}'.format(i,num_tests))
        # Generate a random path
        x_true, y_true, t, vel = rpg.get_harder_path()
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NNControl()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)

        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,net=None)
        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,net=network)
    plt.boxplot(np.divide(nn_fitness,pid_fitness))
    plt.ylabel('Fitness relative to PID')
    plt.show()
    plt.boxplot(np.divide(nn_fitness,pid_fitness),showfliers=False)
    plt.ylabel('Fitness relative to PID')
    plt.show()
    
def trailer_mass_variation_test(network):
    # Initialize memory for fitness evaluation
    
    rpg = RandomPathGenerator()
    # Generate a random path
    x_true, y_true, t, vel = rpg.get_harder_path(vel=25)
    # Set trailer mass alpha values to be looped through
    alpha = np.linspace(0.05,2,num=39)
    pid_fitness = np.zeros(len(alpha))
    nn_fitness = np.zeros(len(alpha))
    for i in range(0,len(alpha)):
        print('{} of {}'.format(i,len(alpha)))
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NNControl()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_pid.modify_parameters(m2_alpha=alpha[i])
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn.modify_parameters(m2_alpha=alpha[i])
        
        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,net=None)
        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,net=network)
        
    plt.plot(alpha,pid_fitness)
    plt.plot(alpha,nn_fitness)
    plt.xlabel('Percentage of design trailer mass (%)')
    plt.ylabel('Fitness')
    plt.ylim(0,1000)
    plt.legend(['PID','Neurocontroller'])
    plt.show()
    
def trailer_stiffness_variation_test(network):
    # Initialize memory for fitness evaluation
    
    rpg = RandomPathGenerator()
    # Generate a random path
    x_true, y_true, t, vel = rpg.get_harder_path(vel=25)
    # Set trailer mass alpha values to be looped through
    alpha = np.linspace(0.05,2,num=39)
    pid_fitness = np.zeros(len(alpha))
    nn_fitness = np.zeros(len(alpha))
    for i in range(0,len(alpha)):
        print('{} of {}'.format(i,len(alpha)))
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NNControl()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_pid.modify_parameters(Ctrailer_alpha=alpha[i])
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn.modify_parameters(Ctrailer_alpha=alpha[i])

        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,net=None)
        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,net=network)
    plt.plot(alpha,pid_fitness)
    plt.plot(alpha,nn_fitness)
    plt.xlabel('Percentage of design trailer tire stiffness (%)')
    plt.ylabel('Fitness')
    plt.ylim(0,1000)
    plt.legend(['PID','Neurocontroller'])
    plt.show()
    
def initial_displacement_test(network):
    Benchmark1=pd.read_csv('Benchmark_DLC_31ms_reduced.csv',sep=',',header=0)
    Benchmark1=Benchmark1.values
    x_true = Benchmark1[:,0]
    y_true = Benchmark1[:,1]
    t = Benchmark1[:,2]
    vel = Benchmark1[:,3]
    
    disp = np.linspace(0,20,num=41)
    
    pid_fitness = np.zeros(len(disp))
    nn_fitness = np.zeros(len(disp))
    for i in range(0,len(disp)):
        print('{} of {}'.format(i,len(disp)))
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NNControl()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)

        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true+disp[i],vel,net=None)
        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true+disp[i],vel,net=network)
    plt.plot(disp,pid_fitness)
    plt.plot(disp,nn_fitness)
    plt.xlabel('Initial lateral displacement from path (m)')
    plt.ylabel('Fitness')
    plt.ylim(0,1000)
    plt.legend(['PID','Neurocontroller'])
    plt.show()
    
def fitness_from_simulation_loop(controller,ego,t,x_true,y_true,vel,net=None):
    # initialize memory
    x = []
    y = []
    delta = []
    th1 = []
    th2 = []
    # Run through control loop for both controllers
    for i in range(0,len(t)):
        state = ego.convert_world_state_to_front()
        if net is not None:
            net = net.float()
            ctrl_delta, ctrl_vel, _,_,_ = controller.calc_steer_control(t[i],state,x_true,y_true, vel, net)
        else:
            ctrl_delta, ctrl_vel, _,_,_ = controller.calc_steer_control(t[i],state,x_true,y_true,vel)
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        
    fitness, _ = calc_off_tracking(x, y, th1, th2, ego.P, x_true, y_true)
    return fitness
        
if __name__ == "__main__":
    trailer_mass_variation_test(2)