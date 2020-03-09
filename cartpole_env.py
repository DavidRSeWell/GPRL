import gym
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
#matplotlib.use('svg')



def H(x):
    '''
    Definition of the hill from GP RL paper
    :param x:
    :return:
    '''

    if x < 0:
        return x**2 + x

    else:
        return x / (np.sqrt(1 + 5*x**2))


def H_Update(x,v,a,dt):
    '''
    velocity = velocity + acceleration
    Position = position + velocity
    :param x: position
    :param v: velocity
    :param a: acceleratioin
    :return:
    '''

    v = v + a*dt
    x = x + v*dt
    return x, v

def test_gym():

    import gym
    env = gym.make('MountainCar-v0')
    #env = gym.wrappers.Monitor(env_to_wrap, '/Users/befeltingu/GPRL/Data/', force=True)
    env.reset()
    for _ in range(10):
        #env.render()
        action_r = 2
        env.step(action_r)  # take a random action

    env.close()
    #env_to_wrap.close()


def stream_app():
    st.title('sup bros')
    st.sidebar.title('Carpole bros with GPs')

    go_button = st.sidebar.button('GO')

    env = gym.make('MountainCar-v0')
    env.reset()


    if go_button:
        video_file = open('/Users/Befeltingu/GPRL/Data/openaigym.video.0.98727.video000000.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)






if __name__ == '__main__':

    plot_H_x = 0
    if plot_H_x:
        x = np.linspace(-1,1,50)

        y = [H(x_i) for x_i in x]

        plt.plot(x,y)

        plt.title('Hill Climb')

        plt.show()

    test_apply_force = 0
    if test_apply_force:
        x = np.linspace(-1, 1, 50)

        x_i = -0.5
        v = 0
        F = -1.0
        dt = 0.3
        x = [x_i]

        for i in range(10):

            x_i, v = H_Update(x_i,v,F,dt)
            F = 0
            x.append(x_i)

        plt.plot([x_j for x_j in x],[t for t in range(-1,10)])

        plt.show()

    test_cart_env = 1
    if test_cart_env:
        test_gym()

    run_streamlit_app = 0
    if run_streamlit_app:
        stream_app()




