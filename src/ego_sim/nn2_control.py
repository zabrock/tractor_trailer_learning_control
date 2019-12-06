#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:05:39 2019

@author: Zeke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:23:05 2019

@author: Zeke
"""
import numpy as np
import torch

class NN2Control(object):
    def __init__(self,
              control_lookahead = 50):
        '''
        Initialization for a neural network controller.
        
        Inputs:
            control_lookahead: Number of points ahead of the previous closest 
                point on the path that the controller looks to find the next
                closest point. Used to handle paths that cross back on themselves.
        '''
        self.err_int = np.zeros(2)
        self.err_d1 = np.zeros(2)
        self.diff_d1 = np.zeros(2)
        self.ctrl_look = control_lookahead
        self.last_closest_idx = 0
        self.t_d1 = 0
        
    def reset(self):
        '''
        Resets the integral and derivative terms
        '''
        self.err_int = np.zeros(2)
        self.err_d1 = np.zeros(2)
        self.diff_d1 = np.zeros(2)
        self.last_closest_idx = 0
        self.t_d1 = 0
        
    def calc_steer_control(self,t,state,path_x,path_y,path_vel,HD2,network,noise=None):
        '''
        Calculates steering control given a path and current state.
        
        Inputs:
            t: Current time value of simulation.
            state: numpy array containing [x_front, y_front, delta, theta1, theta2] which represent, as follows:
                x_front: x-coordinate of current front axle location in world coordinates
                y_front: y-coordinate of current front axle location in world coordinates
                delta: current steer tire angle relative to vehicle (radians)
                theta1: absolute orientation of truck in world coordinates (radians)
                theta2: absolute orientation of trailer in world coordinates (radians)
            path_x: Array of x-coordinates for points that discretize the desired path
            path_y: Array of y-coordinates for points that discretize the desired path
            path_vel: Array of truck velocities desired at each point that discretizes the desired path
            
            Note that path_x, path_y, and path_vel must be the same length for correct functionality.
            
        Returns:
            ctrl_delta: Desired steer tire angle for next time step relative to vehicle (radians)
            ctrl_vel: Desired velocity for next time step
        '''
        # Find index of closest point along path
        # Note that this method assumes positive progress along the path at every time step
        # since it only checks points ahead of the last closest index
        dist_squared = [(state[0]-x)**2 + (state[1]-y)**2 
                  for x,y in zip(path_x[self.last_closest_idx:self.last_closest_idx+self.ctrl_look],
                     path_y[self.last_closest_idx:self.last_closest_idx+self.ctrl_look])]
        
        I_min = self.last_closest_idx + np.argmin(dist_squared)
        # Get the desired velocity at the closest point
        ctrl_vel = path_vel[I_min]
        # Find cross-track and heading error between the current ppsition and desired path
        ct_err, hd_err = calc_path_error(state,path_x,path_y,I_min)
        if noise is not None:
            added_noise = np.random.normal(0,noise/3)
            ct_err = (1+added_noise)*ct_err
            hd_err = (1+added_noise)*hd_err
        err = np.array([ct_err,hd_err])
        # Compute desired steering angle
        Ts = t - self.t_d1
        tau = 0.1 # Time constant for filtering discrete derivatives
        err_diff = ((2*tau-Ts)/(2*tau+Ts))*self.diff_d1 + (2/(2*tau+Ts))*(err-self.err_d1)
        self.err_int += (err+self.err_d1)/2
        stuff1=list(err)
        stuff2=list(err_diff)
        stuff3=list(self.err_int)
        ctrl_delta=network(torch.tensor([float(stuff1[0]),float(stuff1[1]),float(ctrl_vel),float(stuff2[0]),float(stuff3[0]),float(HD2)]))
        ctrl_delta=ctrl_delta.data.numpy()
        ctrl_delta=np.asscalar(ctrl_delta)
        # Limit the steer angle command
        if ctrl_delta > 2*np.pi/5:
            ctrl_delta = 2*np.pi/5
        elif ctrl_delta < -2*np.pi/5:
            ctrl_delta = -2*np.pi/5
            
        # Age the data
        self.t_d1 = t
        self.err_d1 = err
        self.diff_d1 = err_diff
        self.last_closest_idx = I_min
        
        return ctrl_delta, ctrl_vel, err, self.err_int, err_diff
        
def calc_path_error(state,path_x,path_y,I_min):
    '''
    Calculates cross-track and heading error for a given state relative to a path.
    
    Inputs:
        state: numpy array containing [x_front, y_front, delta, theta1, theta2] which represent, as follows:
            x_front: x-coordinate of current front axle location in world coordinates
            y_front: y-coordinate of current front axle location in world coordinates
            delta: current steer tire angle relative to vehicle (radians)
            theta1: absolute orientation of truck in world coordinates (radians)
            theta2: absolute orientation of trailer in world coordinates (radians)
        path_x: Array of x-coordinates for points that discretize the desired path
        path_y: Array of y-coordinates for points that discretize the desired path
        I_min: index of point relative to which error should be calculated; typically,
            the index of the point with the shortest distance to state.
    '''
    # Start by determining closest three points on the desired path curve
    closest_pt = np.array([path_x[I_min],path_y[I_min]])
    if I_min > 0:
        closest_pt_rev = np.array([path_x[I_min-1],path_y[I_min-1]])
    else:
        closest_pt_rev = closest_pt
    if I_min < len(path_x)-1:
        closest_pt_fwd = np.array([path_x[I_min+1],path_y[I_min+1]])
    else:
        closest_pt_fwd = closest_pt
        
    # Get the cross track error by finding minimum distance from the line
    # segments defined by these three points
    ct_error = path_distance(closest_pt_rev, closest_pt, closest_pt_fwd, np.array([state[0],state[1]]))
    
    # Get the heading of the path by taking the arctangent of the vector
    # from the reverse closest point to the forward closest point
    tan_vec = closest_pt_fwd - closest_pt_rev
    path_angle = np.arctan2(tan_vec[1],tan_vec[0])
    heading_error = wrap_to_pi(state[2] + state[3] - path_angle)
    
    return ct_error, heading_error

def calc_trailer_error(state,path_x,path_y):
    '''
    Inputs:
        state: numpy array containing [x_front, y_front, delta, theta1, theta2] which represent, as follows:
            x_front: x-coordinate of current front axle location in world coordinates
            y_front: y-coordinate of current front axle location in world coordinates
            delta: current steer tire angle relative to vehicle (radians)
            theta1: absolute orientation of truck in world coordinates (radians)
            theta2: absolute orientation of trailer in world coordinates (radians)
        path_x: Array of x-coordinates for points that discretize the desired path
        path_y: Array of y-coordinates for points that discretize the desired path
    '''
            
    x_trail = state[0] - 5.51*np.cos(state[3]) - 11.4847*np.cos(state[4])
    y_trail = state[1] - 5.51*np.sin(state[3]) - 11.4847*np.sin(state[4])
    dist_squared = [(x_trail-x)**2 + (y_trail-y)**2 for x,y in zip(path_x, path_y)]
    I_min = np.argmin(dist_squared)
    
    # Start by determining closest three points on the desired path curve
    closest_pt = np.array([path_x[I_min],path_y[I_min]])
    if I_min > 0:
        closest_pt_rev = np.array([path_x[I_min-1],path_y[I_min-1]])
    else:
        closest_pt_rev = closest_pt
    if I_min < len(path_x)-1:
        closest_pt_fwd = np.array([path_x[I_min+1],path_y[I_min+1]])
    else:
        closest_pt_fwd = closest_pt
        
    # Get the cross track error by finding minimum distance from the line
    # segments defined by these three points
    ct_error = path_distance(closest_pt_rev, closest_pt, closest_pt_fwd, np.array([x_trail,y_trail]))
    
    return ct_error
        
    
def path_distance(path_pt_rev, path_pt, path_pt_fwd, cur_point):
    '''
    Calculates the distance from a given point to a path segment discretized as two
    line segments. The segments run from path_pt_rev to path_pt and from path_pt to path_pt_fwd.
    
    Inputs:
        path_pt_rev: Numpy array, first point of first line segment. Must be of shape (2,)
        path_pt: Numpy array, second point of first line segment, first point of second line segment. Must be of shape (2,)
        path_pt_fwd: Numpy array, second point of second line segment. Must be of shape (2,)
        cur_point: Numpy array, coordinates of point where distance from path is measured. Must be of shape (2,)
        
    Output:
        Distance from cur_point to the path segment.
    '''
    # Determine the absolute minimum distance between the path and the 
    # current point based on line segments given by the three points
    dist_rev = minimum_distance(path_pt_rev, path_pt, cur_point)
    dist_fwd = minimum_distance(path_pt, path_pt_fwd, cur_point)
    # Return absolute minimum distance
    if abs(dist_rev) < abs(dist_fwd):
        return dist_rev
    else:
        return dist_fwd
    
def minimum_distance(v,w,p):
    '''
    Calculates the directional distance between a line segment and a point.
    Projects the point onto the line segment and then utilizes the cross product
    to determine whether the distance is "positive" or "negative". Positive values
    indicate the point lies on the "right-hand" side of the vector if the vector 
    is pointing upward; negative values indicate the opposite.
    
    Inputs:
        v: First point of line segment; must be a Numpy array of shape (2,)
        w: Second point of line segment; must be a Numpy array of shape (2,)
        p: Point to which distance is to be calculated; must be a Numpy array of shape (2,)
        
    Output:
        Signed distance from point to line
    '''
    
    # Project the point on the line segment to obtain the projected point
    proj = project_point_on_segment(v,w,p)
    if np.array_equal(v,w):
        return np.sqrt((p[0]-proj[0])**2 + (p[1]-proj[1])**2)
    else:
        # Take the cross product of the vector between the point and the projection and
        # the normalized vector of the line segment; this returns the signed distance
        return np.cross(p-proj, (w-v)/np.linalg.norm(w-v))

def project_point_on_segment(v, w, p):
    '''
    Projects a point p onto the line segment running from point v to point w.
    
    Inputs:
        v: First point of line segment; must be a Numpy array of shape (2,)
        w: Second point of line segment; must be a Numpy array of shape (2,)
        p: Point to which distance is to be calculated; must be a Numpy array of shape (2,)
        
    Output:
        Projection of point p onto line segment vw, as a Numpy array of shape (2,)
    '''
    # Equation for distance from a point to a line segment
    length_sq = (v[0]-w[0])**2 + (v[1]-w[1])**2
    if length_sq == 0:
        return v
    else:
        # Project point on the line given by line segment and restrict
        # the projection to the range [0,1]
        t = np.max([0, np.min([1, np.dot(p-v,w-v)/length_sq])])
        return v + t*(w-v)
        
def wrap_to_pi(angle):
    '''
    Wraps the input angle to the range [-pi, pi]
    
    Inputs:
        angle: Angle to be wrapped to range [-pi, pi]
        
    Output:
        Equivalent angle within range [-pi, pi]
    '''
    wrap = np.remainder(angle, 2*np.pi)
    if abs(wrap) > np.pi:
        wrap -= 2*np.pi * np.sign(wrap)
    return wrap