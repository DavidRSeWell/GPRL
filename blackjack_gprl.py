import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils.utils import set_global_seed

# Make environment
env = rlcard.make('blackjack')
episode_num = 2

# Set a global seed
#set_global_seed(0)

# Set up agents
agent_0 = RandomAgent(action_num=env.action_num)
env.set_agents([agent_0])

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Print out the trajectories
    print('\nEpisode {}'.format(episode))
    for ts in trajectories[0]:
        print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.format(ts[0], ts[1], ts[2], ts[3], ts[4]))


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation of Gaussian Processes in RL by Carl Edward Rasmussen
   http://papers.nips.cc/paper/2420-gaussian-processes-in-reinforcement-learning.pdf
"""

import gym
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

from GP import GP


def k_cov(x1, x2, v, l,sigma):

    A = (v**2)*np.exp(-np.dot((x1 - x2),(x1 - x2))*(1.0 / l**2))

    if np.array_equal(x1,x2):
        return A

    else:
        return A

def dk_dl(x1,x2,v,l):
    return 2*(v**2)*(np.dot((x1 - x2),(x1 - x2))*np.exp(-np.dot((x1 - x2),(x1 - x2))*(1.0 / l**2))) / l**3

def dk_dv(x1,x2,v,l):
    return 2*v*np.exp(-np.dot((x1 - x2),(x1 - x2))*(1.0 / l**2))

def dk_dsigma(x1,x2,sigma):

    if np.array_equal(x1,x2):
        return 2*sigma

    else:
        return 0.0

def sample_discreet_env(env,N):
    '''
    Function to randomly grab samples from
    the environment
    :param env: Gym object
    :return:
    '''

    min_pos = env.min_position
    max_pos = env.max_position
    max_speed = env.max_speed
    goal_position = env.goal_position

    samples = []

    for n in range(N):

        sample_pos = np.random.uniform(min_pos,max_pos)

        sample_vel = np.random.uniform(0,max_speed)

        s = (sample_pos,sample_vel)

        action = np.random.randint(3)

        env.env.state = s # set gym environment to the state to sample from

        env.step(action)

        s_p = env.env.state # new state from the environment

        samples.append((s,action,s_p))


    return samples

def create_train_test(samples):
    '''
    Create train and test sets for
    :param s:
    :return:
    '''
    X = np.zeros((len(samples),3))
    Y = np.zeros((len(samples),2))

    for i in range(len(samples)):
        s, a, s_p = samples[i]
        x_i = np.array([s[0],s[1],a])
        y_i = np.array([s_p[0],s_p[1]])
        X[i] = x_i
        Y[i] = y_i

    return X,Y


class GPRL:
    '''
    This implementation of GPRL is assuming that the algorthim
    has been passed an enivironment obeject of the form used in the
    OPEN AI GYM implementations. I will be using the MountianCar Environment
    '''

    def __init__(self,env,gamma,l,sigma,v):
        '''
        :param env: Gym style environment
        '''
        self.actions = [0,1,2]
        self.env = env
        self.gamma = gamma # future value discount rate
        self.l = l # characteristic length scale
        self.sigma = sigma
        self.v = v # hyper parameter

        #####################
        # Where my GP's at? #
        #####################
        self.GP_V = None # Value GP
        self.GP_E = None # GP for the environment

        ##################
        # Support points #
        ##################
        self.pos_m = np.array([]) # Pos values
        self.vel_m = np.array([]) # velocity values
        self.S = np.array([])
        self.V = np.array([]) # Values lookup Table
        self.W = np.array([]) # User for computing

    def act_greedy(self, s):
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

            self.env.env.state = s

            s_p,r,d,n = self.env.step(a)

            if r < 0: r = 0

            if d == True:
                r = 1

            #x_index = np.where(self.pos_m == x)[0][0]
            #dx_index = np.where(self.vel_m == dx)[0][0]

            s_p = s_p.reshape((1,s_p.shape[0]))
            v_s = self.GP_V.predict(s_p)

            #v_s = np.array(self.GP_V.mean)[0][0]

            a_v = r + self.gamma * v_s

            if a_v > a_best_v:
                a_max = a
                a_best_v = a_v

        return a_max

    def compute_environment_dynamics(self):
        '''
        Train a GP to learn the dynamics
        We actually train 2 GPs one to learn
        position given state and action (s,a).
        The other to learn velocity given (s,a)
        :return:
        '''
        samples_train = sample_discreet_env(env, 500)

        X_train, Y_train = create_train_test(samples_train)

        #samples_test = sample_discreet_env(env, 5)

        #X_test, Y_test = create_train_test(samples_test)

        self.gp_x = GP()

        self.gp_x.train(X_train, Y_train[:,0])

        #rand_index = np.random.randint(0,len(X_test))
        #random_point = X_test[rand_index].reshape((1,3))

        #gp_x.predict(random_point)

        #prediction = gp_x.mean

        # velocity GP
        self.gp_dx = GP()

        self.gp_dx.train(X_train, Y_train[:,1])

        #gp_dx.predict(random_point)

        #prediction_dx = gp_dx.mean

        #actual = Y_test[rand_index]

        print('Done bro')

    def compute_max_marginal(self,GP,y):
        '''
        Maximize marginal likelihood
        :param wrt: The variable to maximize the likelihood w.r.t
        :return:
        '''

        K_inv = np.linalg.inv(GP.cov)

        X = self.S.reshape((self.S.shape[0]**2,2))

        alpha = np.dot(K_inv,(y - GP.mean))

        A = np.dot(alpha,alpha.T) - K_inv

        # I am embarassed about the following code but its too late and im too tired to clean
        l_linespace = np.linspace(0.001,5,10)
        l_min = 999
        best_l = GP.l
        #best_l = 0.50075
        for l_i in l_linespace:

            dk_dl_mat = GP.k_mat(lambda x,y: dk_dl(x, y, GP.v,l_i),X,X)

            min_l = np.array((1.0 / 2) * np.matrix.trace(np.dot(A, dk_dl_mat)))[0][0]

            if min_l < l_min:
                best_l = l_i
                l_min = min_l


        v_linespace = np.linspace(0.001, 2, 5)
        v_min = 999
        best_v = GP.v
        for v_i in v_linespace:

            dk_dv_mat = GP.k_mat(lambda x, y: dk_dv(x, y, v_i, GP.l),X,X)

            min_v = np.array((1.0 / 2) * np.matrix.trace(np.dot(A, dk_dv_mat)))[0][0]

            if min_v < v_min:
                best_v = v_i
                v_min = min_v

        sigma_linespace = np.linspace(0.01, 2, 5)
        sigma_min = 999
        best_sigma = GP.sigma
        '''
        for sigma_i in sigma_linespace:

            dk_dsigma_mat = GP.k_mat(lambda x,y: dk_dsigma(x,y,sigma_i),X,X)

            min_sigma = np.array((1.0 / 2) * np.matrix.trace(np.dot(A, dk_dsigma_mat)))[0][0]

            if min_sigma < sigma_min:
                best_sigma = sigma_i
                sigma_min = min_sigma
        '''

        '''
        I = np.eye(K_inv.shape[0])

        max_l = np.array((1.0 / 2)*np.matrix.trace(np.dot(A,dk_dl_mat)))

        max_v = np.array((1.0 / 2)*np.matrix.trace(np.dot(A,dk_dv_mat)))

        max_sigma = np.array((1.0 / 2)*np.matrix.trace(np.dot(A,2*I)))

        max_l = np.clip(max_l,0.1,8)
        max_v = np.clip(max_l,0.01,2)
        max_sigma = np.clip(max_l,0.01,2)
        '''
        self.GP_V.k_func = lambda x,y: k_cov(x,y,best_v,best_l,best_sigma)

        GP_V.v = best_v
        GP_V.l = best_l
        GP_V.sigma = best_sigma

        print('Max likelihood new params v = {}, l = {}, sigma = {}'.format(GP_V.v,GP_V.l,GP_V.sigma))

        return best_l,best_v,best_sigma

    def compute_W_i(self,i):
        '''
        W is Matrix used in computing Value
        :return:
        '''

        N = self.S.shape[0]**2
        S = self.S.reshape((N,2))

        #for i,s_i in enumerate(S):

        W_i = np.zeros((N,1))

        K_E = self.GP_E.cov[i].T # covariance of the environment at i
        M = np.eye(2)*(self.GP_V.l)**2 # diagnol matrix
        I = np.eye(2)
        #A = (np.linalg.det((np.linalg.inv(M)) * K_E + I))**(-1/2)
        A = (np.linalg.det(I))**(-1/2)
        #B = np.linalg.inv(M + K_E)
        B = np.linalg.inv(M)
        mean_i = self.GP_E.mean[i]

        for j,s_j in enumerate(S):

            diff = (np.reshape(s_j,(s_j.shape[0],1)) - mean_i.T)
            W_i[j] = A*(self.GP_V.v**2)*np.exp((-1.0/2)*np.dot(np.dot(diff.T,B),diff))

        self.W[i] = W_i.reshape((W_i.shape[0],))

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
        self.S = np.zeros((n,n,2)) #TODO Get rid of hardcoded state length 2

        for i, x in enumerate(self.pos_m):
            for j, dx in enumerate(self.vel_m):
                self.S[i,j] = np.array([x,dx])
                self.V[i,j] = self.sample_env(x,dx)

    def init_value(self,m=25):
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
        self.create_grid(m)

    def get_max_derivative(self,dK_dtheta,var_range):
        '''
        Plot the max marginal w.r.t a given variable
        for mulitiple values of that variable
        :param N:
        :return:
        '''
        '''
        m = []

        for v_i in var_range:

            DK_DT = self.GP_V.
            max_l = np.array((1.0 / 2) * np.matrix.trace(np.dot(A, dk_dl_mat)))

            m.append(max_v)

        plt.plot(v,m)
        plt.show()
        '''
        pass

    def plot_value_func(self,V,text=''):
        '''
        Plot the Value matrix in 3D
        :param V: Matrix were rows are position and columns are velocity
        :return:
        '''

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        pos, vel = np.meshgrid(self.pos_m, self.vel_m)

        surf = ax.plot_surface(pos, vel, V.T,cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0, 1)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title(text)
        plt.show()

    def run(self,T=50):
        '''
        Steps of algorithm 1 from paper
            1-2 Modeling System dynamics
            3: Init Value function by sampling m support points
            4: Policy Iteration
        :param T: Number of steps to run per episode
        :return:
        '''

        ######################
        # Compute GP for ENV
        ######################


        self.init_value(m=20)

        self.env.reset()

        s = self.env.env.state

        N = self.V.shape[0]

        self.W = np.zeros((N**2,N**2))

        Y = self.V.reshape((N**2,1)) #init V with R

        S = self.S.reshape((N**2,2))

        self.GP_V.train(S,S,Y)

        #self.GP_V.predict(S)

        V_s = self.GP_V.mean.reshape((N,N))

        self.plot_value_func(V_s,'original V')

        self.compute_max_marginal(self.GP_V,Y)

        self.GP_V.train(S,S, Y)

        #self.GP_V.predict(S)

        V_s = self.GP_V.mean.reshape((N, N))

        self.plot_value_func(V_s,'Max Marginal')

        for t in range(T):

            R = np.zeros((S.shape[0],1))
            V = np.zeros((S.shape[0],1))
            for i , s_i in enumerate(S):

                a = self.act_greedy(s_i)

                self.env.env.state = s_i

                s,r,d,_ = self.env.step(a)

                if r < 0: r = 0

                if d == True:
                    r = 1

                R[i] = r

                s = s.reshape((1,s.shape[0]))

                v_s = self.GP_V.predict(s)

                #v_s = np.array(self.GP_V.mean)[0][0]

                V[i] = r + self.gamma*v_s

                #self.compute_W_i(i)

            # COMPUTE V
            #I = np.eye(np.W.shape[0])
            #V = (I - self.gamma*np.dot(self.W,np.linalg.inv(self.GP_V.cov)))
            #V = np.dot(np.linalg.inv(V),R)

            self.V = V

            #Y = V.reshape((N ** 2,))  # init V with R

            self.GP_V.train(S,S, V)

            #self.GP_V.predict(S)

            self.compute_max_marginal(self.GP_V, V)

            self.GP_V.train(S,S, V)

            #self.GP_V.predict(S)

            V_s = self.GP_V.mean.reshape((N, N))

            self.plot_value_func(V_s,'Value at iteration {}'.format(t))

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

    def simulate_env(self):
        '''
        Run the actual gym enviroment
        to visualize how it performs
        :param N:
        :return:
        '''
        env_wrap = gym.wrappers.Monitor(self.env, '/Users/befeltingu/GPRL/Data/', force=True)

        env_wrap.reset()

        for _ in range(1000):

            env_wrap.render()

            action_r = self.act_greedy(env_wrap.state)

            s,r,d,_ = env_wrap.step(action_r)  # take a random action

            if d:
                break


        self.env.close()
        env_wrap.close()


if __name__ == '__main__':

    #############################
    # GPRL Hyper parameters
    #############################
    T = 20 # number of iterations to run the model

    #############################
    # Value GP Hyper parameters
    #############################
    V_SIGMA = 0.01 # noise
    L = 1 # Length scale
    V = 0.01 #

    env = rlcard.make('blackjack')

    env.reset()

    gprl = GPRL(env,gamma=0.8,l=L,sigma=V_SIGMA,v=V)

    gprl.simulate_env()

    env.close()

    #gprl.compute_environment_dynamics()

    #GP_V = GP(V,L,V_SIGMA,k_func=lambda x,y: k_cov(x, y, V,L,V_SIGMA))

    #gprl.GP_V = GP_V

    #gprl.run(T=T)

    #gprl.simulate_env()
