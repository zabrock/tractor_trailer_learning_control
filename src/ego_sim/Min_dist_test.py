#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:50:41 2019

@author: Zeke
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from ego_sim import EgoSim
from stanley_pid import StanleyPID
from stanley_pid import calc_path_error
from random_path_generator import RandomPathGenerator
import time
import math


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
    names = ['time', 'velocity', 'lateral_accel', 'steer_angle', 'x_front_left', 'x_front_right', 'y_front_left',
             'y_front_right']
    df = pd.read_csv(file, header=0, names=names, usecols=[0, 1, 2, 3, 4, 5, 18, 19])
    x_front = (df['x_front_left'] + df['x_front_right']) / 2
    y_front = (df['y_front_left'] + df['y_front_right']) / 2
    t = df['time']
    vel = df['velocity']
    steer_angle = df['steer_angle'] / 20

    return x_front, y_front, t, vel, steer_angle


def ego_ol_test():
    '''
    Open-loop test of truck-trailer simulation. Takes the data from a Simpack
    CSV and simulates the open-loop output using the Simpack-recorded control
    signals. Plots the resulting driven path against the Simpack-recorded path.
    '''
    x_true, y_true, t, vel, steer_angle = read_path_from_original_simpack_csv(
        '/Users/Zeke/Documents/MATLAB/Model Validation/P4_TCO_Sleeper__FE17_Trailer/DLC/Benchmark_DLC_31ms_pandas.csv')

    ego = EgoSim(sim_timestep=t[1] - t[0], world_state_at_front=True)

    x = []
    y = []
    delta = []
    th1 = []
    th2 = []

    for i in range(0, len(t)):
        xt, yt, deltat, th1t, th2t = ego.simulate_timestep([vel[i], steer_angle[i]])
        x.append(xt)
        y.append(yt)
        delta.append(deltat)
        th1.append(th1t)
        th2.append(th2t)

    plt.plot(x, y)
    plt.plot(x_true, y_true, 'r--')
    plt.show()


def pid_test():
    '''
    Test for the Stanley PID controller. Takes the path data from a Simpack CSV
    and uses it as the desired path for the controller to follow. Plots the
    resulting driven path against the Simpack-recorded path.
    '''
    #	x_true,y_true,t,vel,steer_angle = read_path_from_original_simpack_csv('/Users/Zeke/Documents/MATLAB/Model Validation/P4_TCO_Sleeper__FE17_Trailer/DLC/Benchmark_DLC_31ms_pandas.csv')
    x_true, y_true, t, vel, steer_angle = read_path_from_original_simpack_csv(
        '/Users/Zeke/Documents/MATLAB/Model Validation/P4_TCO_Sleeper__FE17_Trailer/SScorner_var_spd/left/Benchmark_SScorner_80m_left_pandas.csv')

    ego = EgoSim(sim_timestep=t[1] - t[0], world_state_at_front=True)
    pid = StanleyPID()

    x = []
    y = []
    delta = []
    th1 = []
    th2 = []

    for i in range(0, len(t)):
        state = ego.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, ct, hd = pid.calc_steer_control(t[i], state, x_true, y_true, vel)
        xt, yt, deltat, th1t, th2t = ego.simulate_timestep([ctrl_vel, ctrl_delta])
        x.append(xt)
        y.append(yt)
        delta.append(deltat)
        th1.append(th1t)
        th2.append(th2t)

    plt.plot(x, y)
    plt.plot(x_true, y_true, 'r--')
    plt.show()


def random_path_test():
    start_time = time.time()
    rpg = RandomPathGenerator()
    x_true, y_true, t, vel = rpg.get_random_path()
    ego = EgoSim(sim_timestep=t[1] - t[0], world_state_at_front=True)
    pid = StanleyPID()

    x = []
    y = np.zeros(len(t))
    delta = []
    th1 = []
    th2 = []
    ct_err = []
    deltat = 0

    for i in range(0, len(t)):
        xt, yt, er, th1t, th2t = 0, 0, 0, 0, 0
        state = ego.convert_world_state_to_front()
        ctrl_delta, ctrl_vel, ct, hd = pid.calc_steer_control(t[i], state, x_true, y_true, vel)
        xt, yt, er, th1t, th2t = ego.simulate_timestep([ctrl_vel, ctrl_delta])
        x.append(xt)
        y[i] = yt
        delta.append(er)
        th1.append(th1t)
        th2.append(th2t)
        ct_err.append(ct)

    truck_offtrack, trail_offtrack = calc_off_tracking(x, y, th1, th2, ego.P, x_true, y_true)
    #print(ct_err)
    print(truck_offtrack)
    plt.figure(1)
    plt.plot(range(len(ct_err)), ct_err, 'g')
    plt.ylabel("ct_err")
    plt.figure(2)
    plt.plot(range(len(truck_offtrack)), truck_offtrack, 'b')
    plt.ylabel("Off Track error")
    plt.figure(3)
    plt.plot(x_true, y_true, 'r--')
    plt.plot(x, y, 'g--')
    print("Execution time " + str(time.time() - start_time))
    plt.figure(4)
    plt.plot(range(len(trail_offtrack)), trail_offtrack)
    plt.show()


def calc_off_tracking(x_front, y_front, th1, th2, P, path_x, path_y):
    # Preallocate
    start_time = time.time()
    print("In")
    x_c_mat = []
    y_c_mat = []
    x_trail_mat = []
    y_trail_mat = []

    # Calculate the rear and trailer axle positions:
    for i in range(len(x_front)):
        x_c = x_front[i] - (P['l1'] - P['c']) * cos(th1[i])
        x_c_mat.append(x_c)
        y_c = y_front[i] - (P['l1'] - P['c']) * sin(th1[i])
        y_c_mat.append(y_c)
        x_trail = x_c - (P['l2']) * cos(th2[i])
        x_trail_mat.append(x_trail)
        y_trail = y_c - (P['l2']) * sin(th2[i])
        y_trail_mat.append(y_trail)

    truck_mindist_mat = []
    trail_mindist_mat = []

    for j in range(len(x_front)):
        state_1 = [x_front[j], y_front[j], 0, th1[j], th2[j]]
        dist_squared_truck = [(x_front[j] - x) ** 2 + (y_front[j] - y) ** 2
                        for x, y in zip(path_x, path_y)]
        I_min_truck = np.argmin(dist_squared_truck)
        truck_mindist, _ = calc_path_error(state_1, path_x, path_y, I_min_truck)

        state_2 = [x_trail_mat[j], y_trail_mat[j], 0, th1[j], th2[j]]
        dist_squared_trail = [(x_trail_mat[j] - x) ** 2 + (y_trail_mat[j] - y) ** 2
                              for x, y in zip(path_x, path_y)]
        I_min_trail = np.argmin(dist_squared_trail)
        trail_mindist, _ = calc_path_error(state_2, path_x, path_y, I_min_trail)

        truck_mindist_mat.append(truck_mindist)
        trail_mindist_mat.append(trail_mindist)

    print("Off track function time is " + str(time.time()-start_time))
    return truck_mindist_mat, trail_mindist_mat


if __name__ == "__main__":
    #	ego_ol_test()
    #	pid_test()
    random_path_test()
