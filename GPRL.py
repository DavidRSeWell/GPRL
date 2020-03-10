#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of Gaussian Processes in RL by Carl Edward Rasmussen
   http://papers.nips.cc/paper/2420-gaussian-processes-in-reinforcement-learning.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

class GPRL:
    '''
    This implementation of GPRL is assuming that the algorthim
    has been passed an enivironment obeject of the form used in the
    OPEN AI GYM implementations. I will be using the MountianCar Environment
    '''
    def __init__(self,env,gamma,l):
        '''
        :param env: Gym style environment
        '''
        self.actions = [0,1,2]
        self.env = env
        self.gamma = gamma # future value discount rate
        self.l = l # characteristic length scale


        ##################
        # Support points #
        ##################
        self.pos_m = np.array([]) # Pos values
        self.vel_m = np.array([]) # velocity values
        self.V = np.array([]) # Values lookup Table

    def create_grid(self,n=25):
        '''
        Create 2d grid of position,velocity values
        :param n:
        :return:
        '''
        min_pos = self.env.min_position
        max_pos = self.env.max_position
        max_speed = self.env.max_speed

        self.pos_m = np.linspace(min_pos,max_pos,n)
        self.vel_m = np.linspace(-max_speed,max_speed,n)

        self.V = np.zeros((n,n))

        for i, x in enumerate(self.pos_m):
            for j, dx in enumerate(self.vel_m):
                self.V[i,j] = self.sample_env(x,dx)

    def act_greedy(self,s):
        '''
        From current state
        get max action
        :param s:
        :return:
        '''

        self.env.reset()
        a_max = None
        a_best_v = -999
        for a in self.actions:

            self.env.env = s

            self.env.step(a)

            x, dx = self.env.state

            a_v = self.sample_env(x,dx) + self.gamma*self.V[x,dx]

            if a_v > a_best_v:
                a_max = a
                a_best_v = a_v

        return a

    def init_value(self,m=50):
        '''
        Sample m support vectors and initialize
        V (m x d) = (V(s_1) ... V(s_m)) where V(s_i) = R_i
        Where R_i is given by the environment or sampled
        from the system dynamics and then computed using equation (7) from paper
        :param m: Number of support vectors
        :return:
        '''

        # For now just creating a grid of values initilized to
        # the reward at that point in the environment
        self.create_grid()

    def plot_value_func(self):

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        pos, vel = np.meshgrid(self.pos_m, self.vel_m)

        surf = ax.plot_surface(pos, vel, self.V.T,cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def plot_value_func_2d(self):

        plt.plot

    def run(self,T=50):
        '''
        Steps of algorithm 1 from paper
            1-2 Modeling System dynamics
            3: Init Value function by sampling m support points
            4: Policy Iteration
        :param T: Number of steps to run per episode
        :return:
        '''
        self.init_value()

        self.env.reset()

        s = self.env.env.state

        for t in range(T):

            a = self.act_greedy(s)

            s,r,d = self.env.step(a)

            if d:
                break

    def sample_env(self,p,v):
        '''
        :param p:
        :param v:
        :return:
        '''

        r1 = bool(p >= self.env.goal_position and v >= self.env.goal_velocity)

        return r1

    def sample_discreet_env(self,M):
        '''
        Function to randomly grab samples from
        the environment
        :param env: Gym object
        :return:
        '''

        min_pos = self.env.min_position
        max_pos = self.env.max_position
        max_speed = self.env.max_speed
        goal_position = self.env.goal_position
        goal_velocity = self.env.goal_velocity

        samples = []

        for n in range(M):
            sample_pos = np.random.uniform(min_pos, max_pos)

            sample_vel = np.random.uniform(0, max_speed)

            s = (sample_pos, sample_vel)

            action = np.random.randint(3)

            self.env.env.state = s  # set gym environment to the state to sample from

            r1 = bool(sample_pos >= goal_position and sample_vel >= goal_velocity)

            self.env.step(action)

            s_p = self.env.env.state  # new state from the environment

            r2 = bool(s_p[0] >= goal_position and s_p[1] >= goal_velocity)

            samples.append((s, r1, action, s_p,r2))

        return samples


