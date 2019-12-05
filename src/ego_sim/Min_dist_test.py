#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:50:41 2019

@author: Zeke
"""
import numpy as np
from numpy import cos, sin
from stanley_pid import calc_path_error
#import time
import matplotlib.pyplot as plt

def calc_off_tracking(x_front, y_front, th1, th2, P, path_x, path_y):
    # Preallocate
#    start_time = time.time()
#    print("In")
    x_c_mat = []
    y_c_mat = []
    x_trail_mat = []
    y_trail_mat = []
#
    # Calculate the rear and trailer axle positions:
#    for i in range(len(x_front)):
#        x_c = x_front[i] - (P['l1'] - P['c']) * cos(th1[i])
#        x_c_mat.append(x_c)
#        y_c = y_front[i] - (P['l1'] - P['c']) * sin(th1[i])
#        y_c_mat.append(y_c)
#        x_trail = x_c - (P['l2']) * cos(th2[i])
#        x_trail_mat.append(x_trail)
#        y_trail = y_c - (P['l2']) * sin(th2[i])
#        y_trail_mat.append(y_trail)

    truck_mindist_mat = []
    trail_mindist_mat = []

    for j in range(len(x_front)):
        state_1 = [x_front[j], y_front[j], 0, th1[j], th2[j]]
        dist_squared_truck = [(x_front[j] - x) ** 2 + (y_front[j] - y) ** 2
                        for x, y in zip(path_x, path_y)]
        I_min_truck = np.argmin(dist_squared_truck)
        truck_mindist, _ = calc_path_error(state_1, path_x, path_y, I_min_truck)

        x_trail = x_front[j] - (P['l1'] - P['c']) * cos(th1[j]) - (P['l2']) * cos(th2[j])
        y_trail = y_front[j] - (P['l1'] - P['c']) * sin(th1[j]) - (P['l2']) * sin(th2[j])
        state_2 = [x_trail, y_trail, 0, th1[j], th2[j]]
        dist_squared_trail = [(x_trail - x) ** 2 + (y_trail - y) ** 2
                              for x, y in zip(path_x, path_y)]
        I_min_trail = np.argmin(dist_squared_trail)
        trail_mindist, _ = calc_path_error(state_2, path_x, path_y, I_min_trail)

        truck_mindist_mat.append(truck_mindist)
        trail_mindist_mat.append(trail_mindist)

    err_truck = np.square(truck_mindist_mat)
    sqrd_err_truck = np.sum(err_truck)

    err_trail = np.square(trail_mindist_mat)
    sqrd_err_trail = np.sum(err_trail)
    
#    plt.plot(x_front,y_front)
#    plt.plot(x_trail_mat,y_trail_mat)
#    plt.show()
#    plt.plot(np.arange(0,len(x_front)),truck_mindist_mat)
#    plt.plot(np.arange(0,len(x_front)),trail_mindist_mat)
#    plt.show()
    
    return sqrd_err_truck + sqrd_err_trail, err_trail
