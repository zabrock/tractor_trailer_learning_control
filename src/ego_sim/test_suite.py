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
import pandas as pd
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def basic_fitness_comparison(network,num_tests=10):
    # Initialize memory for fitness evaluation
    pid_fitness = np.zeros(num_tests)
    nn_fitness = 200*np.random.random(num_tests)
    rpg = RandomPathGenerator()
    for i in range(0,num_tests):
        print('{} of {}'.format(i,num_tests))
        # Generate a random path
        x_true, y_true, t, vel = rpg.get_harder_path(end_time=4)
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NN2Control()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,net=None)
        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,net=network)
    plt.boxplot(np.divide(nn_fitness,pid_fitness))
    plt.ylabel('Fitness relative to PID\n(lower is better)')
    plt.xticks([1],['Neurocontroller'],)
    plt.show()
    # Plot without outliers
    plt.boxplot(np.divide(nn_fitness,pid_fitness),showfliers=False)
    plt.ylabel('Fitness relative to PID\n(lower is better)')
    plt.xticks([1],['Neurocontroller'])
    plt.show()
    
def trailer_mass_variation_test(network,num_tests=5):
    # Initialize memory for fitness evaluation
    
    rpg = RandomPathGenerator()
    # Generate a random path
    xs = []
    ys = []
    ts = []
    vels = []
    for i in range(0,num_tests):
        x1,y1,t1,v1 = rpg.get_harder_path(end_time=10,vel=25)
        xs.append(x1)
        ys.append(y1)
        ts.append(t1)
        vels.append(v1)
    # Set trailer mass alpha values to be looped through
    alpha = np.linspace(0.1,2,num=20)
    pid_fitness = []
    nn_fitness = []
    print('new')
    for i in range(0,len(alpha)):
        print('{} of {}'.format(i,len(alpha)))
        
        for j in range(0,num_tests):
            x_true = xs[j]
            y_true = ys[j]
            t = ts[j]
            vel = vels[j]
            # Start a new PID controller and ego_sims for both controllers
            pid = StanleyPID()
            nn = NN2Control()
            ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            ego_pid.modify_parameters(m2_alpha=alpha[i])
            ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            ego_nn.modify_parameters(m2_alpha=alpha[i])
            pid_fitness.append(fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,net=None))
            nn_fitness.append(fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,net=network))
        
    pid_fitness_avg = [np.mean(fitness) for fitness in pid_fitness]
    nn_fitness_avg = [np.mean(fitness) for fitness in nn_fitness]
    pid_fitness_std = [np.std(fitness) for fitness in pid_fitness]
    nn_fitness_std = [np.std(fitness) for fitness in nn_fitness]
    plt.errorbar(alpha*100,pid_fitness_avg,yerr=pid_fitness_std,marker='s',capsize=5)
    plt.errorbar(alpha*100,nn_fitness_avg,yerr=nn_fitness_std,marker='s',capsize=5)
    plt.xlabel('Percentage of design trailer mass (%)')
    plt.ylabel('Sum squared tracking error, $m^2$\n(lower is better)')
#    plt.ylim(0,2000)
    plt.legend(['PID','Neurocontroller'])
    plt.show()
    
def trailer_stiffness_variation_test(network,num_tests=5):
    # Initialize memory for fitness evaluation
    
    rpg = RandomPathGenerator()
    # Generate a random path
    xs = []
    ys = []
    ts = []
    vels = []
    for i in range(0,num_tests):
        x1,y1,t1,v1 = rpg.get_harder_path(end_time=10,vel=25)
        xs.append(x1)
        ys.append(y1)
        ts.append(t1)
        vels.append(v1)
        
    # Set trailer mass alpha values to be looped through
    alpha = np.linspace(0.1,2,num=20)
    pid_fitness = []
    nn_fitness = []
    for i in range(0,len(alpha)):
        print('{} of {}'.format(i,len(alpha)))

        for j in range(0,num_tests):
            x_true = xs[j]
            y_true = ys[j]
            t = ts[j]
            vel = vels[j]
            # Start a new PID controller and ego_sims for both controllers
            pid = StanleyPID()
            nn = NN2Control()
            ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            ego_pid.modify_parameters(Ctrailer_alpha=alpha[i])
            ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            ego_nn.modify_parameters(Ctrailer_alpha=alpha[i])
            pid_fitness.append(fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,net=None))
            nn_fitness.append(fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,net=network))
        
    pid_fitness_avg = [np.mean(fitness) for fitness in pid_fitness]
    nn_fitness_avg = [np.mean(fitness) for fitness in nn_fitness]
    pid_fitness_std = [np.std(fitness) for fitness in pid_fitness]
    nn_fitness_std = [np.std(fitness) for fitness in nn_fitness]
    plt.errorbar(alpha*100,pid_fitness_avg,yerr=pid_fitness_std,marker='s',capsize=5)
    plt.errorbar(alpha*100,nn_fitness_avg,yerr=nn_fitness_std,marker='s',capsize=5)
    plt.xlabel('Percentage of design trailer tire stiffness (%)')
    plt.ylabel('Sum squared tracking error, $m^2$\n(lower is better)')
    plt.ylim(0,1.1*(max(pid_fitness_avg)+max(pid_fitness_std)))
    plt.legend(['PID','Neurocontroller'])
    plt.show()
    
def trailer_length_variation_test(network,num_tests=5):
    rpg = RandomPathGenerator()
    # Generate a random path
    xs = []
    ys = []
    ts = []
    vels = []
    for i in range(0,num_tests):
        x1,y1,t1,v1 = rpg.get_harder_path(end_time=10,vel=25)
        xs.append(x1)
        ys.append(y1)
        ts.append(t1)
        vels.append(v1)
    alpha = np.linspace(0.1,2,num=20)
    
    pid_fitness = []
    nn_fitness = []
    print('new')
    for i in range(0,len(alpha)):
        print('{} of {}'.format(i,len(alpha)))
        
        for j in range(0,num_tests):
            x_true = xs[j]
            y_true = ys[j]
            t = ts[j]
            vel = vels[j]
            # Start a new PID controller and ego_sims for both controllers
            pid = StanleyPID()
            nn = NN2Control()
            ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            ego_pid.modify_parameters(Ctrailer_alpha=alpha[i])
            ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            ego_nn.modify_parameters(Ctrailer_alpha=alpha[i])
            pid_fitness.append(fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,net=None))
            nn_fitness.append(fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,net=network))
        
    pid_fitness_avg = [np.mean(fitness) for fitness in pid_fitness]
    nn_fitness_avg = [np.mean(fitness) for fitness in nn_fitness]
    pid_fitness_std = [np.std(fitness) for fitness in pid_fitness]
    nn_fitness_std = [np.std(fitness) for fitness in nn_fitness]
    plt.errorbar(alpha*100,pid_fitness_avg,yerr=pid_fitness_std,marker='s',capsize=5)
    plt.errorbar(alpha*100,nn_fitness_avg,yerr=nn_fitness_std,marker='s',capsize=5)
    plt.xlabel('Percentage of design trailer length (m)')
    plt.ylabel('Sum squared tracking error, $m^2$\n(lower is better)')
    plt.ylim(0,1.1*(max(pid_fitness_avg)+max(pid_fitness_std)))
    plt.legend(['PID','Neurocontroller'])
    plt.show()
    
def initial_displacement_test(network):
    Benchmark1=pd.read_csv('Benchmark_DLC_31ms_reduced.csv',sep=',',header=0)
    Benchmark1=Benchmark1.values
    x_true = Benchmark1[:,0]
    y_true = Benchmark1[:,1]
    t = Benchmark1[:,2]
    vel = Benchmark1[:,3]
    
    disp = np.linspace(0,20,num=40)
    
    pid_fitness = np.zeros(len(disp))
    nn_fitness = np.zeros(len(disp))
    for i in range(0,len(disp)):
        print('{} of {}'.format(i,len(disp)))
        # Start a new PID controller and ego_sims for both controllers
        pid = StanleyPID()
        nn = NN2Control()
        ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
        ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)

        pid_fitness[i] = fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true+disp[i],vel,net=None)
        nn_fitness[i] = fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true+disp[i],vel,net=network)
    plt.plot(disp,pid_fitness)
    plt.plot(disp,nn_fitness)
    plt.xlabel('Initial lateral displacement from path (m)')
    plt.ylabel('Sum squared tracking error, $m^2$\n(lower is better)')
    plt.ylim(0,1.1*max(pid_fitness))
    plt.legend(['PID','Neurocontroller'])
    plt.show()
    
def noisy_signal_test(network,num_tests=5):
    rpg = RandomPathGenerator()
    # Generate a random path
    xs = []
    ys = []
    ts = []
    vels = []
    for i in range(0,num_tests):
        x1,y1,t1,v1 = rpg.get_harder_path(end_time=10,vel=25)
        xs.append(x1)
        ys.append(y1)
        ts.append(t1)
        vels.append(v1)
    
    noise = np.linspace(0,0.2,num=20)
    
    pid_fitness = []
    nn_fitness = []
    for i in range(0,len(noise)):
        print('{} of {}'.format(i,len(noise)))
        
        for j in range(0,num_tests):
            x_true = xs[j]
            y_true = ys[j]
            t = ts[j]
            vel = vels[j]
            # Start a new PID controller and ego_sims for both controllers
            pid = StanleyPID()
            nn = NN2Control()
            ego_pid = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            ego_nn = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
    
            pid_fitness.append(fitness_from_simulation_loop(pid,ego_pid,t,x_true,y_true,vel,net=None,noise=noise[i]))
            nn_fitness.append(fitness_from_simulation_loop(nn,ego_nn,t,x_true,y_true,vel,net=network,noise=noise[i]))
        
    pid_fitness_avg = [np.mean(fitness) for fitness in pid_fitness]
    nn_fitness_avg = [np.mean(fitness) for fitness in nn_fitness]
    pid_fitness_std = [np.std(fitness) for fitness in pid_fitness]
    nn_fitness_std = [np.std(fitness) for fitness in nn_fitness]
    plt.errorbar(noise*100,pid_fitness_avg,yerr=pid_fitness_std,marker='s',capsize=5)
    plt.errorbar(noise*100,nn_fitness_avg,yerr=nn_fitness_std,marker='s',capsize=5)
    plt.xlabel('Percent noise in error signals (%)')
    plt.ylabel('Sum squared tracking error, $m^2$\n(lower is better)')
    plt.ylim(0,1.1*(max(pid_fitness_avg)+max(pid_fitness_std)))
    plt.legend(['PID','Neurocontroller'])
    plt.show()
    
def fitness_from_simulation_loop(controller,ego,t,x_true,y_true,vel,net=None,noise=None):
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
            ctrl_delta, ctrl_vel, _,_,_ = controller.calc_steer_control(t[i],state,x_true,y_true, vel, state[4]-state[3], net, noise=noise)
        else:
            ctrl_delta, ctrl_vel, _,_,_ = controller.calc_steer_control(t[i],state,x_true,y_true,vel, noise=noise)
#        print(ctrl_delta)
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        
    plt.plot(x,y)
    if net is not None:
        plt.title('Network')
    else:
        plt.title('PID')
    plt.show()
    fitness, _ = calc_off_tracking(x, y, th1, th2, ego.P, x_true, y_true)
    return fitness
        
if __name__ == "__main__":
    trailer_mass_variation_test(2)