#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:34:22 2019

@author: Zeke
"""
from random_path_generator import RandomPathGenerator
from ego_sim import EgoSim
from stanley_pid import StanleyPID
from nn2_control import NN2Control
import numpy as np
import matplotlib.pyplot as plt
from Min_dist_test import calc_off_tracking

def basic_fitness_comparison(network,num_tests=10):
    # Initialize memory for fitness evaluation
    pid_fitness = np.zeros(num_tests)
    nn_fitness = 200*np.random.random(num_tests)
    rpg = RandomPathGenerator()
    for i in range(0,num_tests):
        # Generate a random path
        x_true, y_true, t, vel = rpg.get_harder_path()
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NN2Control()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)

        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,network=None)
#        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,network=network)
    plt.boxplot(np.divide(nn_fitness,pid_fitness))
    
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
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NN2Control()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_pid.modify_parameters(m2_alpha=alpha[i])
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn.modify_parameters(m2_alpha=alpha[i])
        
        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,network=None)
#        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,network=network)
        
    plt.plot(alpha,pid_fitness)
#    plt.plot(alpha,nn_fitness)
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
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NN2Control()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_pid.modify_parameters(Ctrailer_alpha=alpha[i])
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn.modify_parameters(Ctrailer_alpha=alpha[i])

        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,network=None)
#        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,network=network)
    plt.plot(alpha,pid_fitness)
#    plt.plot(alpha,nn_fitness)
    plt.show()
    
def fitness_from_simulation_loop(controller,ego,t,x_true,y_true,vel,network=None):
    # initialize memory
    x = []
    y = []
    delta = []
    th1 = []
    th2 = []
    # Run through control loop for both controllers
    for i in range(0,len(t)):
        state = ego.convert_world_state_to_front()
        if network is not None:
            ctrl_delta, ctrl_vel, _,_,_ = controller.calc_steer_control(t[i],state,x_true,y_true,vel,network)
        else:
            ctrl_delta, ctrl_vel, _,_,_ = controller.calc_steer_control(t[i],state,x_true,y_true,vel)
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        
    fitness, _ = calc_off_tracking(x, y, th1, th2, ego.P, x_true, y_true)
    return fitness
        
if __name__ == "__main__":
    trailer_mass_variation_test(2)