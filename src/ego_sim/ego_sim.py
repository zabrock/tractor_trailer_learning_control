#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:43:57 2019

@author: Zeke
"""
import numpy as np
from scipy.integrate import odeint
from scipy import interpolate
import copy

class EgoSim(object):
    def __init__(self,sim_timestep=0.02,world_state_at_front=False):
        '''
        Initialization function. EgoSim simulates at constant timesteps given by
        sim_timestep. 
        
        Inputs:
            sim_timestep: Simulation timestep to be used, in seconds
            world_state_at_front: dictates whether the simulation output state
                is given in coordinates at the truck's front axle (if True) or 
                at the truck's center of gravity (if False).
        '''
        self.init = False
        self.set_default_truck_params()
        self.world_state = np.zeros(5)
        self.truck_state = np.zeros(4)
        self.M = self.calculate_mass_matrix()
        self.B = np.array([self.P['C1'],self.P['a1']*self.P['C1'],0,0])
        self.sim_time = 0
        self.sim_timestep = sim_timestep
        self.world_state_at_front = world_state_at_front
        
    def set_default_truck_params(self):
        '''
        Default parameters to be used for the simulation. Run at initialization.
        If other parameters are desired, the corresponding entries in the P
        dictionary for the EgoSim object may be overwritten.
        '''
        P = {'m1': 9159.63, # Mass of the truck, kg
               'I1': 55660.3, # Inertia of the truck, kg*m^2
               'm2': 27091.8, # Mass of the trailer, kg
               'I2': 386841, # INertia of the trailer, kg*m^2
               'a1': 2.43264, # Distance from truck COG to front axle, m
               'c': 0.2286, # Firth-wheel offset from truck drive axle virtual center, m
               'l1': 5.7404, # Wheelbase from truck front axle to drive axle virtual center, m
               'l2': 16.104-0.914-3.083-1.2446/2, # Distance from trailer axle to fifth wheel, m
               'a2': 5.9336, # Distance from fifth wheel to trailer COG, m
               'h1': 3.0792, # Distance from truck COG to fifth wheel, m
               'b1': 3.3078, # Distance from truck COG to drive axle virtual center, m
               'b2': 5.5511, # Distance from trailer COG to trailer axle, m
               'C1': 187020.2*2, # Sum of truck front axle cornering stiffness, N/rad
               'C2': 130274.0*4, # Sum of truck's first drive axle cornering stiffness, N/rad
               'C3': 130274.0*4, # Sum of truck's second drive axle cornering stiffness, N/rad
               'C4': 115000.0*4, # Sum of trailer's first axle cornering stiffness, N/rad
               'C5': 115000.0*4, # Sum of trailer's second drive axle cornering stiffness, N/rad
               # The following parameters are used for visualization or more advanced collision checking only
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
        
    def modify_parameters(self,m1_alpha=1,m2_alpha=1,Csteer_alpha=1,Cdrive_alpha=1,Ctrailer_alpha=1,l2_alpha=1):
        '''
        Modifies loading conditions and tire stiffness for the truck and trailer.
        '''
        self.P['m1'] = m1_alpha*9159.63
        self.P['I1'] = m1_alpha*55660.3
        self.P['m2'] = m2_alpha*27091.8
        self.P['I2'] = m2_alpha*386841
        self.P['C1'] = Csteer_alpha*187020.2*2
        self.P['C2'] = Cdrive_alpha*130274.0*4
        self.P['C3'] = Cdrive_alpha*130274.0*4
        self.P['C4'] = Ctrailer_alpha*115000.0*4
        self.P['C5'] = Ctrailer_alpha*115000.0*4
        self.P['l2'] = l2_alpha*(16.104-0.914-3.083-1.2446/2)
        self.P['a2'] = l2_alpha*5.9336
        self.P['b2'] = l2_alpha*5.5511
        
        
    def calculate_mass_matrix(self):
        '''
        Calculates the mass matrix as defined in the Luijten lateral dynamic model.
        
        Output:
            Mass matrix for linearized dynamic model, as a Numpy matrix.
        '''
        P = self.P
        M = np.matrix([[P['m1']+P['m2'], -P['m2']*(P['h1']+P['a2']),-P['m2']*P['a2'],0],\
                  [-P['m2']*P['h1'], P['I1']+P['m2']*P['h1']*(P['h1']+P['a2']), P['m2']*P['h1']*P['a2'], 0],\
                  [-P['m2']*P['a2'], P['I2']+P['m2']*P['a2']*(P['h1']+P['a2']), P['I2']+P['m2']*P['a2']**2, 0],\
                  [0, 0, 0, 1]])
        return M
    
    def simulate_timestep(self,ctrl):
        '''
        Simulates the truck-trailer system for a single timestep, using the saved
        truck state as the initial condition.
        
        Inputs:
            ctrl: Numpy array of shape (2,) with the control velocity in the first index
                and control steer tire angle (radians) in the second.
                
        Outputs:
            Truck's state in world coordinates, either at the front axle (if
                self.truck_state_at_front is True) or at the truck center of gravity
                (if self.truck_state_at_front is False)
        '''
        # Craft the stiffness matrix
        A = self.calculate_stiffness_matrix(ctrl)
        # Define start and end solve time for this timestep
        t = np.array([self.sim_time, self.sim_time + self.sim_timestep])
        # Solve the system for this timestep
        self.truck_state = odeint(canonical_ode,self.truck_state,t,args=(self.M,A,self.B,ctrl[1]))[-1]
        # Update the world state with the new truck state and update sim time
        self.update_world_state(ctrl)
        self.sim_time = t[-1]
        
        # Output the results either at the front axle coordinates or the truck C.O.G. coordinates.
        if self.world_state_at_front:
            return self.convert_world_state_to_front()
        else:
            return self.world_state
        
    def calculate_stiffness_matrix(self,ctrl):
        '''
        Calculates the stiffness matrix A from the Luijten dynamic model equation
        Mx' = Ax + Bu.
        
        Inputs:
            ctrl: Numpy array of shape (2,) with the control velocity in the first index
                and control steer angle (radians) in the second. 
                
        Outputs:
            Stiffness matrix A, as a Numpy matrix.
        '''
        # First element of control signal is longitudinal velocity
        u1 = ctrl[0]
        P = self.P
        # Following calculations come from the Luijten dynamic model
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
        '''
        Updates the world state by integrating the values found in truck state.
        This function should be run once and only once whenever the truck state is updated
        in simulation.
        
        Inputs:
            ctrl: Numpy array of shape (2,) with the control velocity in the first index
                and control steer angle (radians) in the second. 
        '''
        # self.world_state is [x, y, delta, theta1, theta2]
        # self.truck_state is [v1, theta1dot, phidot, phi]
        # ctrl is [u1, delta]
        
        # Integrate theta1dot
        self.world_state[3] += self.truck_state[1]*self.sim_timestep
        
        # Rotate truck-frame velocity into the world frame
        vel = np.array([ctrl[0],self.truck_state[0]]) # [u1, v1]
        vel = np.matmul(self.rotation_matrix(self.world_state[3]),vel)
        # Travel along the velocity vector for the time step
        self.world_state[0:2] += vel.A1[0:2]*self.sim_timestep
        
        # Calculate absolute orientation of the trailer
        self.world_state[4] = self.world_state[3] + self.truck_state[3]
        
        # Output steer angle
        self.world_state[2] = ctrl[1]
        self.world_state[3] = wrap_to_pi(self.world_state[3])
        self.world_state[4] = wrap_to_pi(self.world_state[4])
        self.truck_state[3]=wrap_to_pi(self.truck_state[3])
        
    def convert_world_state_to_front(self):
        '''
        Outputs the world state with the truck coordinate frame placed on the front
        axle rather than the truck's center of gravity. Does not modify the
        world state in memory.
        '''
        # Move along vector from x and y coordinates of world_state in direction of theta
        state = copy.deepcopy(self.world_state)
        state[0] += self.P['a1']*np.cos(state[3])
        state[1] += self.P['a1']*np.sin(state[3])
        return state
        
    def rotation_matrix(self,theta):
        '''
        Generates and returns a 2D roration matrix for the angle theta.
        '''
        return np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
def canonical_ode(y, t, M, A, B, u):
    '''
    Canonical ODE for My' = Ay + Bu. Used for ODE solver.
    '''
    dydt = np.matmul(np.linalg.inv(M)*A,y) + np.matmul(np.linalg.inv(M),B)*u
    return dydt.A1

def sinusoid_input():
    ego = EgoSim()
    x = []
    y = []
    delta = []
    th1 = []
    th2 = []
    t = np.arange(0,20,step=0.02)
    ctrl_vel = 31
    for i in range(0,len(t)):
        ctrl_delta = 2*np.pi*np.sin(t[i]*3.5)/5
        xt,yt,deltat,th1t,th2t = ego.simulate_timestep([ctrl_vel,ctrl_delta])
        x.append(xt); y.append(yt); delta.append(deltat); th1.append(th1t); th2.append(th2t)
        
    
    return x, y

def wrap_to_pi(angle):
    '''
    Wraps the input angle to the range [-pi, pi]
    
    Inputs:
        angle: Angle to be wrapped to range [-pi, pi]
        
    Output:
        Equivalent angle within range [-pi, pi]
    '''
    #print(angle)
    wrap = np.remainder(angle, 2*np.pi)
    if abs(wrap) > np.pi:
        wrap -= 2*np.pi * np.sign(wrap)
    return wrap

if __name__ == "__main__":
    x, y = sinusoid_input()
    import matplotlib.pyplot as plt
    plt.plot(x,y)
    plt.gca().set_aspect('equal', adjustable='box')
    
    import pandas as pd
    df = pd.DataFrame({'x':x,'y':y})
    df.to_csv('test_path.csv')
    