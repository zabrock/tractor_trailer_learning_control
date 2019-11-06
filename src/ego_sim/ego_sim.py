#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:43:57 2019

@author: Zeke
"""
import numpy as np
from scipy.integrate import odeint

class EgoSim(object):
	def __init__(self,sim_timestep=0.02):
		self.init = False
		self.set_default_truck_params()
		self.world_state = np.zeros(5)
		self.truck_state = np.zeros(4)
		self.M = self.calculate_mass_matrix()
		self.B = np.array([self.P['C1'],self.P['a1']*self.P['C1'],0,0])
		self.sim_time = 0
		self.sim_timestep = sim_timestep
		
	def set_default_truck_params(self):
		P = {'m1': 9159.63,
			   'I1': 55660.3,
			   'm2': 27091.8,
			   'I2': 386841,
			   'a1': 2.43264,
			   'c': 0.2286,
			   'l1': 5.7404,
			   'l2': 16.104-0.914-3.083-1.2446/2,
			   'a2': 5.9336,
			   'h1': 3.0792,
			   'b1': 3.3078,
			   'b2': 5.5511,
			   'C1': 187020.2*2,
			   'C2': 130274.0*4,
			   'C3': 130274.0*4,
			   'C4': 115000.0*4,
			   'C5': 115000.0*4,
			   'truck_width': 2.57,
			   'truck_str_ax2front': 1.295,
			   'truck_str_ax2rear': 7.275,
			   'trailer_width': 2.59,
			   'trailer_5th2front': 0.914,
			   'trailer_5th2rear': 16.104
			   }
		P['Cs1'] = P['a1']*P['C1'] - P['b1']*(P['C2']+P['C3'])
		P['Cq1'] = P['a1']**2*P['C1'] + P['b1']**2*(P['C2']+P['C3'])
		self.P = P
		
	def calculate_mass_matrix(self):
		P = self.P
		M = np.matrix([[P['m1']+P['m2'], -P['m2']*(P['h1']+P['a2']),-P['m2']*P['a2'],0],\
				  [-P['m2']*P['h1'], P['I1']+P['m2']*P['h1']*(P['h1']+P['a2']), P['m2']*P['h1']*P['a2'], 0],\
				  [-P['m2']*P['a2'], P['I2']+P['m2']*P['a2']*(P['h1']+P['a2']), P['I2']+P['m2']*P['a2']**2, 0],\
				  [0, 0, 0, 1]])
		return M
	
	def simulate_timestep(self,ctrl):
		# Craft the stiffness matrix
		A = self.calculate_stiffness_matrix(ctrl)
		t = np.array([self.sim_time, self.sim_time + self.sim_timestep])
		self.truck_state = odeint(canonical_ode,self.truck_state,t,args=(self.M,A,self.B,ctrl[1]))[-1]
		self.update_world_state(ctrl)
		print(self.truck_state)
		self.sim_time = t[-1]
		return self.world_state
		
	def calculate_stiffness_matrix(self,ctrl):
		# First element of control signal is longitudinal velocity
		u1 = ctrl[0]
		P = self.P
		a11 = P['C1'] + P['C2'] + P['C3'] + P['C4'] + P['C5']
		a12 = P['Cs1'] - (P['C4']+P['C5'])*(P['h1']+P['l2']) + (P['m1']+P['m2'])*u1**2
		a13 = -(P['C4']+P['C5'])*P['l2']
		a14 = -(P['C4']+P['C5'])*u1
		a21 = P['Cs1'] - (P['C4']+P['C5'])*P['h1']
		a22 = P['Cq1'] + (P['C4']+P['C5'])*(P['h1']+P['l2'])*P['h1'] - P['m2']*P['h1']*u1**2
		a23 = (P['C4']+P['C5'])*P['h1']*P['l2']
		a24 = (P['C4']+P['C5'])*P['h1']*u1
		a31 = -(P['C4']+P['C5'])*P['l2']
		a32 = (P['C4']+P['C5'])*P['l2']*(P['h1']+P['l2']) - P['m2']*P['a2']*u1**2
		a33 = (P['C4']+P['C5'])*P['l2']**2
		a34 = (P['C4']+P['C5'])*P['l2']*u1
		return -(1/u1)*np.matrix([[a11, a12, a13, a14],\
							   [a21, a22, a23, a24],\
							   [a31, a32, a33, a34],\
							   [0, 0, -u1, 0]])
		
	
	def update_world_state(self,ctrl):
		# self.world_state is [x, y, delta, theta1, theta2]
		# self.truck_state is [v1, theta1dot, phidot, phi]
		# ctrl is [u1, delta]
		
		# Integrate theta1dot
		self.world_state[3] += self.truck_state[1]*self.sim_timestep
		
		# Rotate truck-frame velocity into the world frame
		vel = np.array([ctrl[0],self.truck_state[0]]) # [u1, v1]
		print(vel)
		vel = np.matmul(self.rotation_matrix(self.world_state[3]),vel)
		# Travel along the velocity vector for the time step
		self.world_state[0:2] += vel.A1[0:2]*self.sim_timestep
		
		# Calculate absolute orientation of the trailer
		self.world_state[4] = self.world_state[3] + self.truck_state[3]
		
	def rotation_matrix(self,theta):
		return np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	
def canonical_ode(y, t, M, A, B, u):
	dydt = np.matmul(np.linalg.inv(M)*A,y) + np.matmul(np.linalg.inv(M),B)*u
	return dydt.A1