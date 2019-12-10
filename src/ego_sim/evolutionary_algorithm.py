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
import pickle
import matplotlib.pyplot as  plt


class EvolutionaryAlgorithm(object):
    def __init__(self,nn_controller,pop_size=10,pct_weight_variation=0.2):
        # Save number of controllers to keep through each iteration
        #print('bias value in start of evo',nn_controller.fc3.bias.data)
        self.pop_size = pop_size
        self.pid_fitness=0
        # Save the percent weight variation to use when permutating controllers
        self.pct_weight_var = pct_weight_variation
        nn_controller=nn_controller.float()
        temp_controllers = pickle.loads(pickle.dumps(nn_controller))
        #print('bias value in temp of evo',temp_controllers.fc3.bias.data)
        # Initialize population of controllers randomly perturbed from the input controller
        self.controllers=[nn_controller,nn_controller,nn_controller,nn_controller,nn_controller]
        for i in range(4):
            self.controllers[i] = self.permutate_controller(temp_controllers)
        
        #for i in  range(len(self.controllers)):
            #print('bias value in end of evo',self.controllers[i].fc3.bias.data)
        fitnesses = self.evaluate_fitness()
        
        # Save the best controller's index
        self.best_controller_idx = np.argmin(fitnesses)
    
        
    def permutate_controller(self,nn_controller_orig):
        '''
        Modifies the weights in nn_controller randomly to create a new controller
        '''
        #modify the weights of the 1st layer
        #nn_controller2=nn_controller.deepcopy()
        #print('nn_weight data',nn_controller.fc1.weight.data)
        #print('nn_weight data len',len(nn_controller.fc1.weight.data))
        #print('nn_weight data',nn_controller.fc1.weight.data)
        nn_controller= pickle.loads(pickle.dumps(nn_controller_orig))
        nn_controller=nn_controller.float()
        #print('start permute: ',nn_controller.fc3.bias.data)
        temp=list(nn_controller.fc1.weight.data.shape)
        weight_add=torch.rand(temp)

        weight_add=(-0.5+weight_add)/np.linalg.norm(weight_add)

        nn_controller.fc1.weight.data=nn_controller.fc1.weight.data+weight_add*5
        #modify the weights of the 2nd layer
        temp=list(nn_controller.fc2.weight.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/np.linalg.norm(weight_add)
        nn_controller.fc2.weight.data=nn_controller.fc2.weight.data+weight_add*5
        #modify the weights of the 3rd layer
        temp=list(nn_controller.fc3.weight.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/np.linalg.norm(weight_add)
        nn_controller.fc3.weight.data=nn_controller.fc3.weight.data+weight_add*2.5
        #modify the biases of the 1st layer
        temp=list(nn_controller.fc1.bias.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/np.linalg.norm(weight_add)
        nn_controller.fc1.bias.data=nn_controller.fc1.bias.data+weight_add*1
        #modify the biases of the 2nd layer
        temp=list(nn_controller.fc2.bias.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)/np.linalg.norm(weight_add)
        nn_controller.fc2.bias.data=nn_controller.fc2.bias.data+weight_add*1
        #modify the biases of the 3rd layer
        temp=list(nn_controller.fc3.bias.data.shape)
        weight_add=torch.rand(temp)
        weight_add=(-0.5+weight_add)
        nn_controller.fc3.bias.data=nn_controller.fc3.bias.data+weight_add*0.2
        #print('end permute: ',nn_controller.fc3.bias.data)
        
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
            #print(t)
            #print(x_true)
            #print(y_true)
            #print(vel)    
            ego=EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
            controller = NN2Control()
            #print('controller: ', i)
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
                    #print(ctrl_delta)
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
            #print('pid:',self.pid_fitness)
            #print('controller',self.controller_fitness[i])
            #plt.plot(x_truck,y_truck)
            #plt.plot(x_true,y_true,'--r')
            #plt.show()
            
        
    
    def iterate(self,epsilon=0.1):
        # Pick a network to modify using the epsilon-greedy method

            
        # Randomly modify the network parameters and add it to the pool
        for i in  range(10):
            prob = np.random.random()
            if prob > epsilon:
                new_ctrlr = pickle.loads(pickle.dumps(self.controllers[self.best_controller_idx]))
            else:
                new_ctrlr = pickle.loads(pickle.dumps(random.choice(self.controllers)))
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
        #print('network fitness: ',self.controller_fitness)
        #print('best PID fitness:      ',self.pid_fitness)
        #print(self.best_controller_idx)
        
    def select_next_generation(self):
        '''
        Selects next generation of controllers based on fitness
        '''
        best_controller=np.argsort(self.controller_fitness)
        new_controllers=copy.deepcopy(self.controllers[0:10])
        new_controller_fitness=copy.deepcopy(self.controller_fitness[0:10])
        for i in range(10):
            new_controllers[i]=self.controllers[best_controller[i]]
            new_controller_fitness[i]=self.controller_fitness[best_controller[i]]
        self.controllers=new_controllers
        self.controller_fitness=new_controller_fitness
        
