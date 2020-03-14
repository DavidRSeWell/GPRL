import gym
import numpy as np



class BlackJack:
    '''
    A wrapper of blackjack from https://github.com/datamllab/rlcard/
    Adding some methods to make sampling easier
    '''
    def __init__(self,env=None):
        self.env = env


    def init_env(self):
        self.env = gym.make('Blackjack-v0')


    def create_train_data(self,N):

        samples,ace_samples = self.sample_trajectories(N)

        X = []
        for s,action,r,next_state in samples:
            X.append([s[0],s[1],s[2],action,])

    def sample_trajectories(self,N):
        '''
        Function to assist in sampling from
        the environment
        :return:
        '''

        non_ace_samples = []
        ace_samples = []
        for episode in range(N):

            s = self.env.reset()
            # Generate data from the environment
            while True:

                action = np.random.randint(2)

                next_state, reward, done, info = self.env.step(action)

                print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.format(s, action, reward, next_state,done))

                s = next_state

                if s[2]:

                    ace_samples.append((s,action,reward,next_state))
                else:
                    non_ace_samples.append((s,action,reward,next_state))


                if done: break

        return non_ace_samples,ace_samples




if __name__ == '__main__':


    test_bj_sampling = 1
    if test_bj_sampling:

        bj = BlackJack()
        bj.init_env()
        bj.sample_trajectories(10)
