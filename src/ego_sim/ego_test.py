#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:00:12 2019

@author: Zeke
"""

from ego_sim import EgoSim
import matplotlib.pyplot as plt
import numpy as np

ego = EgoSim(world_state_at_front=True)

x = []
y = []
delta = []
th1 = []
th2 = []

for i in range(0,500):
	xt,yt,deltat,th1t,th2t = ego.simulate_timestep([10,0.1])
	x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
	
t = ego.sim_timestep*np.arange(0,500)

plt.plot(t,x)
plt.show()
plt.plot(t,y)
plt.show()
plt.plot(t,delta)
plt.show()
plt.plot(t,th1)
plt.show()
plt.plot(t,th2)
plt.show()
plt.plot(x,y)
plt.show()