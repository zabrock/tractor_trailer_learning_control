#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:50:41 2019

@author: Zeke
"""
import pandas as pd
import matplotlib.pyplot as plt
from ego_sim import EgoSim
from stanley_pid import StanleyPID

def read_path_from_original_simpack_csv(file):
	'''
	Gets the time, truck velocity, lateral acceleration, steer angle, and
	front axle path coordinates from a Simpack CSV output.
	
	Inputs:
		file: filepath to Simpack-output CSV.
		
	Outputs:
		x_front: Truck front axle path x-coordinates.
		y_front: Truck front axle path y-coordinates.
		t: Vector of time values at which data was recorded.
		vel: Velocity of truck.
		steer_angle: Steer tire angle of truck.
	'''
	names = ['time','velocity','lateral_accel','steer_angle','x_front_left','x_front_right','y_front_left','y_front_right']
	df = pd.read_csv(file,header=0,names=names,usecols=[0,1,2,3,4,5,18,19])
	x_front = (df['x_front_left'] + df['x_front_right'])/2
	y_front = (df['y_front_left'] + df['y_front_right'])/2
	t = df['time']
	vel = df['velocity']
	steer_angle = df['steer_angle']/20
	
	return x_front, y_front, t, vel, steer_angle

def ego_ol_test():
	'''
	Open-loop test of truck-trailer simulation. Takes the data from a Simpack
	CSV and simulates the open-loop output using the Simpack-recorded control
	signals. Plots the resulting driven path against the Simpack-recorded path.
	'''
	x_true,y_true,t,vel,steer_angle = read_path_from_original_simpack_csv('/Users/Zeke/Documents/MATLAB/Model Validation/P4_TCO_Sleeper__FE17_Trailer/DLC/Benchmark_DLC_31ms_pandas.csv')
	
	ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)

	x = []
	y = []
	delta = []
	th1 = []
	th2 = []
	
	for i in range(0,len(t)):
		xt,yt,deltat,th1t,th2t = ego.simulate_timestep([vel[i],steer_angle[i]])
		x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
	
	
	plt.plot(x,y)
	plt.plot(x_true,y_true,'r--')
	plt.show()
	
def pid_test():
	'''
	Test for the Stanley PID controller. Takes the path data from a Simpack CSV
	and uses it as the desired path for the controller to follow. Plots the 
	resulting driven path against the Simpack-recorded path.
	'''
#	x_true,y_true,t,vel,steer_angle = read_path_from_original_simpack_csv('/Users/Zeke/Documents/MATLAB/Model Validation/P4_TCO_Sleeper__FE17_Trailer/DLC/Benchmark_DLC_31ms_pandas.csv')
	x_true,y_true,t,vel,steer_angle = read_path_from_original_simpack_csv('/Users/Zeke/Documents/MATLAB/Model Validation/P4_TCO_Sleeper__FE17_Trailer/SScorner_var_spd/left/Benchmark_SScorner_80m_left_pandas.csv')
	
	ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
	pid = StanleyPID()

	x = []
	y = []
	delta = []
	th1 = []
	th2 = []
	
	for i in range(0,len(t)):
		state = ego.convert_world_state_to_front()
		ctrl_delta, ctrl_vel = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
		xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
		x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
	
	plt.plot(x,y)
	plt.plot(x_true,y_true,'r--')
	plt.show()
	
	
if __name__ == "__main__":
	ego_ol_test()
	pid_test()