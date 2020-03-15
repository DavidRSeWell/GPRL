

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

    def __init__(self,env,gamma):
        '''
        :param env: Gym style environment
        '''
        self.actions = [0,1,2]
        self.env = env
        self.gamma = gamma # future value discount rate

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

            s_p = s_p.reshape((1,s_p.shape[0]))
            v_s = self.GP_V.predict(s_p)[0][0]

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

        V = np.zeros((n,n))
        S = np.zeros((n,n,2)) #TODO Get rid of hardcoded state length 2

        for i, x in enumerate(self.pos_m):
            for j, dx in enumerate(self.vel_m):
                S[i,j] = np.array([x,dx])
                V[i,j] = self.sample_env(x,dx)

        return S,V

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
        S,V = self.create_grid(m)
        self.S = S
        self.V = V

    def plot_best_path(self,iter):
        '''
        Function for taking the current value function
        and plotting a graph of the path taken through
        space.
        :return:
        '''

        path = self.simulate_env()
        y = [x for x in range(1,len(path) + 1)]
        #xs = np.linspace(self.env.min_position, self.env.max_position, 100)
        #ys = self.height(xs)
        #plt.plot(xs,ys)

        plt.plot(path,y)
        plt.scatter(path,y)
        plt.xlim(self.env.min_position,self.env.max_position)
        plt.title('Best path at iteratioi {}'.format(iter))
        plt.show()

    def plot_value_func(self,V,text=''):
        '''
        Plot the Value matrix in 3D
        :param V: Matrix were rows are position and columns are velocity
        :return:
        '''

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        min_pos = self.env.min_position
        max_pos = self.env.max_position
        max_speed = self.env.max_speed

        n = V.shape[0]
        self.pos_m = np.linspace(min_pos, max_pos, n)
        self.vel_m = np.linspace(-max_speed, max_speed, n)

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

        N = self.V.shape[0]

        self.W = np.zeros((N**2,N**2))

        Y = self.V.reshape((N**2,1)) #init V with R

        S = self.S.reshape((N**2,2))

        GRID_SIZE = 100
        S_Grid,_ = self.create_grid(GRID_SIZE)

        self.GP_V = self.GP_V.fit(S, Y)

        S_Grid = S_Grid.reshape((GRID_SIZE**2,2))

        y_pred = self.GP_V.predict(S_Grid)

        V_s = y_pred.reshape((GRID_SIZE, GRID_SIZE))

        self.plot_value_func(V_s,'Max Marginal')


        for t in range(T):

            self.plot_best_path(t)

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

                v_s = self.GP_V.predict(s)[0][0]

                V[i] = r + self.gamma*v_s


            self.V = V.reshape((N,N))

            self.GP_V = self.GP_V.fit(S, V)

            y_pred = self.GP_V.predict(S_Grid)

            V_s = y_pred.reshape((GRID_SIZE, GRID_SIZE))

            self.plot_value_func(V_s, 'Value at iteration {}'.format(t))

        self.plot_value_func(V_s,'Value at iteration {}'.format('Final'))

        self.plot_best_path('Final')

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

    def sample_env(self, p, v):
        '''
        :param p:
        :param v:
        :return:
        '''

        r1 = bool(p >= self.env.goal_position and v >= self.env.goal_velocity)

        return r1

    def simulate_env(self):
        '''
        Run the actual gym enviroment
        to visualize how it performs
        :param N:
        :return:
        '''
        env_wrap = gym.wrappers.Monitor(self.env, '/Users/befeltingu/GPRL/Data/', force=True)

        env_wrap.reset()

        pos_x = []

        for _ in range(200):

            #env_wrap.render()

            action_r = self.act_greedy(env_wrap.state)

            s,r,d,_ = env_wrap.step(action_r)  # take a random action

            pos_x.append(s[0])
            if d:
                break


        self.env.close()
        env_wrap.close()

        return pos_x


if __name__ == '__main__':

    #############################
    # GPRL Hyper parameters
    #############################
    T =  50# number of iterations to run the model

    from gym import envs
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel ,Matern,WhiteKernel

    print(envs.registry.all())

    #############################
    # Value GP Hyper parameters
    #############################
    V_SIGMA = 0.01 # noise
    L = 1 # Length scale
    V = 0.1 #

    env = gym.make('MountainCar-v0')
    env.reset()

    gprl = GPRL(env,gamma=0.8)

    #kernel = RBF(10, (1e-2, 1e2))

    kernel = ConstantKernel() + Matern(length_scale=2, nu=3 / 2) + WhiteKernel(noise_level=1)

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

    gprl.GP_V = gp

    gprl.run(T=T)

    gprl.simulate_env()

