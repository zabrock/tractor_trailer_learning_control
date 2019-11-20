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
from random_path_generator import RandomPathGenerator
import numpy as np
	
def random_path_pid(output_to_csv=True,csv_filename='random_path_pid.csv'):
	rpg = RandomPathGenerator()
	x_true, y_true, t, vel = rpg.get_random_path()
	ego = EgoSim(sim_timestep = t[1]-t[0], world_state_at_front=True)
	pid = StanleyPID()

	ctrl_delta_out, ctrl_vel_out, ct_err, hd_err, ct_err_int, hd_err_int, \
	ct_err_diff, hd_err_diff = [], [], [], [], [], [], [], []
	
	for i in range(0,len(t)):
		state = ego.convert_world_state_to_front()
		ctrl_delta, ctrl_vel, err, err_int, err_diff = pid.calc_steer_control(t[i],state,x_true,y_true, vel)
		ego.simulate_timestep([ctrl_vel,ctrl_delta])
		if not np.isnan(err[0]):
			ctrl_delta_out.append(ctrl_delta); ctrl_vel_out.append(ctrl_vel)
			ct_err.append(err[0]); hd_err.append(err[1])
			ct_err_int.append(err_int[0]); hd_err_int.append(err_int[1])
			ct_err_diff.append(err_diff[0]); hd_err_diff.append(err_diff[1])
		
	csv_output = {'Cross-Track Error (m)':ct_err,
			   'Heading Error (rad)':hd_err,
			   'Desired Velocity (m/s)':ctrl_vel_out,
			   'Command Steer Angle (rad)':ctrl_delta_out,
			   'Integral of CT Error':ct_err_int,
			   'Integral of Heading Error':hd_err_int,
			   'Derivative of CT Error':ct_err_diff,
			   'Derivative of Heading Error':hd_err_diff}
	df = pd.DataFrame(csv_output)
	df.to_csv(csv_filename,index=False)
	
def generate_n_pid_csvs(n=10,csv_filename_base='random_path_pid'):
	for i in range(0,n):
		csv_name = csv_filename_base + '_' + str(i) + '.csv'
		random_path_pid(csv_filename=csv_name)
	
if __name__ == "__main__":
	generate_n_pid_csvs(n=50,csv_filename_base='random_path_pid_more_output')