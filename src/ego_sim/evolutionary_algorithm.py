#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:32:01 2019

@author: Zeke
"""
import numpy as np
import copy
from stanley_pid import StanleyPID
from ego_sim import EgoSim
from random_path_generator import RandomPathGenerator
import random

class EvolutionaryAlgorithm(object):
	def __init__(self,nn_controller,pop_size=10,pct_weight_variation=0.2):
		
		# Initialize population of controllers randomly perturbed from the input controller
		self.controllers = [self.permutate_controller(nn_controller) for i in range(0,pop_size-1)]
		self.controllers.append(nn_controller)
		fitnesses = self.evaluate_fitness()
		
		# Save the best controller's index
		self.best_controller_idx = np.argmin(fitnesses)
		
		# Save number of controllers to keep through each iteration
		self.pop_size = pop_size
		# Save the percent weight variation to use when permutating controllers
		self.pct_weight_var = pct_weight_variation
		
	def permutate_controller(self,nn_controller):
		'''
		Modifies the weights in nn_controller randomly to create a new controller
		'''
		pass
	
	def evaluate_fitness(self):
		'''
		Evaluates and returns the fitness of all controllers in pool
		'''
		pass
	
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
		self.fitnesses.append(0)
		# Evaluate fitness of all controllers on a randomly generated path
		fitnesses = self.evaluate_fitness()
		
		self.update_best_controller(fitnesses)
		# Select next generation from pool
		self.select_next_generation()
		
		
	def update_best_controller(self,fitnesses):
		self.best_controller_idx = np.argmin(fitnesses)
		
	def select_next_generation(self):
		'''
		Selects next generation of controllers based on fitness
		'''