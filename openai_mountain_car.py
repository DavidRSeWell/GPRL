import gym
import numpy as np

from GP import GP
from GPRL import GPRL



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


if __name__ == '__main__':

    print('RUnning GPRL')

    env = gym.make('MountainCar-v0')

    env.reset()

    test_sample_env = 0
    if test_sample_env:

        samples_train = sample_discreet_env(env,50)

        X_train, Y_train = create_train_test(samples_train)

        samples_test = sample_discreet_env(env,100)

        X_test, Y_test = create_train_test(samples_test)

        env.close()

        gp_1 = GP()
        gp_1.train(X_train,Y_train[:,0])
        gp_1.predict(X_test)
        #gp_2 = GP()
        #gp_2.train(X_train,Y_train)

        print('Done running GPRL')

    test_gprl = 0
    if test_gprl:

        gprl = GPRL(env=env,gamma=0.8)

        gprl.init_value(50)

        gprl.plot_value_func()

