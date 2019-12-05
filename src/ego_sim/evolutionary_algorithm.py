#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:32:01 2019

@author: Zeke
"""
import numpy as np
import copy
import torch
from stanley_pid import StanleyPID
from ego_sim import EgoSim
from nn2_control import NN2Control
from random_path_generator import RandomPathGenerator
from Min_dist_test import calc_off_tracking
import random
import copy

class EvolutionaryAlgorithm(object):
    def __init__(self,nn_controller,pop_size=10,pct_weight_variation=0.00000000002):
        # Save number of controllers to keep through each iteration
        self.pop_size = pop_size
        # Save the percent weight variation to use when permutating controllers
        self.pct_weight_var = pct_weight_variation
        nn_controller=nn_controller.float()
        temp_controllers=copy.deepcopy(nn_controller)
        # Initialize population of controllers randomly perturbed from the input controller
        self.controllers = [self.permutate_controller(temp_controllers) for i in range(0,pop_size-1)]
        self.controllers.append(nn_controller)
        fitnesses = self.evaluate_fitness()
        self.pid_fitness=0
        # Save the best controller's index
        self.best_controller_idx = np.argmin(fitnesses)
    
        
    def permutate_controller(self,nn_controller):
        '''
        Modifies the weights in nn_controller randomly to create a new controller
        '''
        #modify the weights of the 1st layer
        #nn_controller2=nn_controller.deepcopy()
        #print('nn_weight data',nn_controller.fc1.weight.data)
        #print('nn_weight data len',len(nn_controller.fc1.weight.data))
        #print('nn_weight data',nn_controller.fc1.weight.data)
        nn_controller=nn_controller.float()
        
        temp=list(nn_controller.fc1.weight.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/sum(weight_add)
        weight_add[:,7]=weight_add[:,7]*10
        nn_controller.fc1.weight.data=nn_controller.fc1.weight.data+weight_add*self.pct_weight_var
        #modify the weights of the 2nd layer
        temp=list(nn_controller.fc2.weight.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/sum(weight_add)
        nn_controller.fc2.weight.data=nn_controller.fc2.weight.data+weight_add*self.pct_weight_var
        #modify the weights of the 3rd layer
        temp=list(nn_controller.fc3.weight.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/sum(weight_add)
        nn_controller.fc3.weight.data=nn_controller.fc3.weight.data+weight_add*self.pct_weight_var
        #modify the biases of the 1st layer
        temp=list(nn_controller.fc1.bias.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/sum(weight_add)
        nn_controller.fc1.bias.data=nn_controller.fc1.bias.data+weight_add*self.pct_weight_var
        #modify the biases of the 2nd layer
        temp=list(nn_controller.fc2.bias.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/sum(weight_add)
        nn_controller.fc2.bias.data=nn_controller.fc2.bias.data+weight_add*self.pct_weight_var
        #modify the biases of the 3rd layer
        temp=list(nn_controller.fc3.bias.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/sum(weight_add)
        nn_controller.fc3.bias.data=nn_controller.fc3.bias.data+weight_add*self.pct_weight_var
        
        
        return nn_controller
    
    def evaluate_fitness(self):
        '''
        Evaluates and returns the fitness of all controllers in pool
        '''
        
        rpg=RandomPathGenerator()
        controller = NN2Control()
        x_true, y_true, t, vel=rpg.get_harder_path(end_time=10)
        x_truck=[]
        y_truck=[]
        self.controller_fitness=np.zeros(len(self.controllers))
        th1t=0
        th2t=0
        th1=[]
        th2=[]
        pid=StanleyPID()
        for i in range(len(self.controllers)+1):
            ego=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            print('controller: ', i)
            th1t=0
            th2t=0
            th1=[]
            th2=[]
            x_truck=[]
            y_truck=[]
            for j in range(len(t)):
                if i == len(self.controllers):
                    state = ego.convert_world_state_to_front()
                    ctrl_delta, ctrl_vel, err, interr, differr = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
                    xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                    x_truck.append(xt)
                    y_truck.append(yt)
                    th1.append(th1t)
                    th2.append(th2t)
                else:
                    state = ego.convert_world_state_to_front()
                    ctrl_delta, ctrl_vel, err, interr, differr = controller.calc_steer_control(t[i],state,x_true,y_true, vel, th1t-th2t, self.controllers[i])
                    xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
                    x_truck.append(xt)
                    y_truck.append(yt)
                    th1.append(th1t)
                    th2.append(th2t)
                #inputs=np.concatenate((err,ctrl_vel,interr,differr),axis=None)
                #network_input=torch.tensor(inputs)
                #out=self.controllers[i](network_input)
                #x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
            if i == len(self.controllers):
                self.pid_fitness, CTerr =calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
            else:
                self.controller_fitness[i], CTerr = calc_off_tracking(x_truck, y_truck, th1, th2, ego.P, x_true, y_true)
        
    
    def iterate(self,epsilon=0.1):
        # Pick a network to modify using the epsilon-greedy method
        prob = np.random.random()
        if prob > epsilon:
            new_ctrlr = copy.deepcopy(self.controllers[self.best_controller_idx])
        else:
            new_ctrlr = copy.deepcopy(random.choice(self.controllers))
            
        # Randomly modify the network parameters and add it to the pool
        new_ctrlr = self.permutate_controller(new_ctrlr)
        self.controllers.append(new_ctrlr)
        #print(self.controller_fitness)
        self.controller_fitness=np.append(self.controller_fitness,0)
        # Evaluate fitness of all controllers on a randomly generated path
        self.evaluate_fitness()
        # Select next generation from pool
        self.select_next_generation()
        self.update_best_controller()

        
        
    def update_best_controller(self):
        self.best_controller_idx = np.argmin(self.controller_fitness)
        print('network fitness: ',self.controller_fitness)
        print('best PID fitness:      ',self.pid_fitness)
        #print(self.best_controller_idx)
        
    def select_next_generation(self):
        '''
        Selects next generation of controllers based on fitness
        '''
        worst_controller=np.argmax(self.controller_fitness)
        self.controllers.pop(worst_controller)
        self.controller_fitness=np.delete(self.controller_fitness,worst_controller)
        
        
