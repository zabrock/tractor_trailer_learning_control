#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:32:11 2019

@author: Zeke
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class RandomPathGenerator(object):
	def __init__(self):
		'''
		Initialization function for random path generator.
		'''
		# Set lateral acceleration limits for the path that will be generated
		self.set_lateral_accel_limits()
		
	def set_lateral_accel_limits(self):
		'''
		Define lateral acceleration limits that the random path should adhere to.
		'''
		self.accel_lim_x = [0,1,3,5,7,11,18,23,25,27,31,33,35]
		self.accel_lim_y = [0,0.25,1.3,3,4.5,5,5.2,4.5,4.6,4.6,4.2,4.2,4]
		
	def plot_lateral_accel_limits(self):
		'''
		Plots the lateral acceleration limits that the random path adheres to.
		'''
		plt.fill_between(self.accel_lim_x,self.accel_lim_y,-np.array(self.accel_lim_y),color='lightseagreen')
		plt.plot(self.accel_lim_x,self.accel_lim_y,'r')
		plt.plot(self.accel_lim_x,-np.array(self.accel_lim_y),'r')
		legend_elements = [Line2D([0], [0], color='r', lw=4, label='Stability limits'),
					 Patch(facecolor='lightseagreen', label='Stable region')]
		plt.legend(handles=legend_elements,loc="upper right")
		plt.xlabel('Truck velocity (m/s)')
		plt.ylabel('Maximum lateral acceleration ($m/s^2$)')
		plt.show()
		
	def get_random_path(self, end_time=40, delta_t=0.02, vel=None):
		'''
		Generates a random path beginning at [0, 0] with initial heading of 0 degrees
		which is traveled at constant velocity vel for the timespan [0, end_time].
		
		Inputs:
			end_time: Seconds over which random path should be traveled at constant velocity.
			delta_t: Time delta between sample points.
			vel: Velocity at which the path should be traversed. If not given,
				randomly picked from set [1,31].
				
		Outputs:
			x: x-coordinates of path
			y: y-coordinates of path
		'''
		# Randomize velocity if not given
		if vel is None:
			vel = 30*np.random.random() + 1
		# Generate time vector of solutions
		t = np.linspace(0,end_time,num=int(end_time/delta_t)+1)
		# Preallocate output arrays
		x = np.zeros(len(t))
		y = np.zeros(len(t))
		# Set initial lateral acceleration to zero
		lat_accel = 0
		# Set initial velocity to be in the x-direction
		vel_vec = np.array([vel, 0.0])
		# Determine the lateral acceleration limit for this velocity
		lat_accel_limit = np.interp(vel,self.accel_lim_x,self.accel_lim_y)
		
		for i in range(0,len(t)-1):
			# Randomize jerk of current time step
			width = 3
#			jerk = np.random.normal(0,2)
			jerk = width*np.random.random() - width/2
			# Integrate to get lateral acceleration
			lat_accel += jerk*delta_t
			# Make sure the lateral acceleration fits within the limits for the
			# velocity we're traveling at
			if abs(lat_accel) > lat_accel_limit:
				lat_accel = lat_accel_limit*np.sign(lat_accel)
			# Find unit normal vector to the current velocity vector
			normal_vec = np.flip(vel_vec)/np.linalg.norm(vel_vec)
			# Add lateral velocity given by lateral acceleration to the velocity
			# vector and scale to maintain magnitude of velocity as constant
			vel_vec += lat_accel*delta_t*normal_vec
			vel_vec = vel*vel_vec/np.linalg.norm(vel_vec)
			
			# Travel along the velocity vector for the time step to determine
			# the next coordinates
			x[i+1] = x[i] + vel_vec[0]*delta_t
			y[i+1] = y[i] + vel_vec[1]*delta_t
			
		return x, y, t, vel*np.ones(len(t))
			
if __name__ == "__main__":
	rpg = RandomPathGenerator()
	x, y, t, v = rpg.get_random_path()
	plt.plot(x, y)
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')
	plt.title('Random path, velocity = {0:.2f} m/s'.format(v[0]))
	plt.show()
	rpg.plot_lateral_accel_limits()