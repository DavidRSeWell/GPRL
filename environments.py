import gym
import numpy as np



class BlackJack:
    '''
    A wrapper of blackjack from https://github.com/datamllab/rlcard/
    Adding some methods to make sampling easier
    '''
    def __init__(self,env=None):
        self.env = env
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def create_environment_train_data(self, N):

        samples, ace_samples = self.sample_trajectories(N)

        X = []
        for s, action, r, next_state in samples:
            next_useable_ace = next_state[2]
            if next_useable_ace: next_useable_ace = 1
            else: next_useable_ace = 0
            X.append([s[0],0, action, next_state[0],next_useable_ace])

        X_ace = []
        for s, action, r, next_state in ace_samples:
            next_useable_ace = next_state[2]
            if next_useable_ace:
                next_useable_ace = 1
            else:
                next_useable_ace = 0
            X_ace.append([s[0],1, action, next_state[0],next_useable_ace])

        X = np.array(X)
        X_ace = np.array(X_ace)

        return X, X_ace

        print('done creating training data')

    def get_state(self):
        return (self.env.player.sum(),self.env.dealer[0],self.env.natural)

    def init_env(self):
        self.env = gym.make('Blackjack-v0')

    def reset(self):
        return self.env.reset()

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

                if s[2]:
                    ace_samples.append((s,action,reward,next_state))
                else:
                    non_ace_samples.append((s,action,reward,next_state))

                s = next_state


                if done: break

        return non_ace_samples,ace_samples

    def draw_card(self):

        return int(np.random.choice(self.deck))

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def usable_ace(self,hand):  # Does this hand have a usable ace?
        return 1 in hand and sum(hand) + 10 <= 21

    def sum_hand(self,hand):  # Return current hand total
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def is_bust(self,hand):  # Is this hand a bust?
        return self.sum_hand(hand) > 21

    def score(self,hand):  # What is the score of this hand (0 if bust)
        return 0 if self.is_bust(hand) else self.sum_hand(hand)

    def is_natural(self,hand):  # Is this hand a natural blackjack?
        return sorted(hand) == [1, 10]



if __name__ == '__main__':


    test_bj_sampling = 1
    if test_bj_sampling:

        bj = BlackJack()
        bj.init_env()
        #bj.sample_trajectories(1000)
        X, X_ace = bj.create_environment_train_data(100)

        print('')
