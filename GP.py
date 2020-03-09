import matplotlib.pyplot as plt
import numpy as np


class GP:

    def __init__(self,k_func=None,sigma=0.5):
        self.k_func = k_func
        self.sigma = sigma

        # Holder variables
        self.X = None
        self.X_star = None
        self.Y = None
        self.noise = None

        # covariance matrices
        self.K_x_x = None
        self.K_x_x_inv = None
        self.K_s_s = None
        self.K_s_x = None

    def k_gauss(self,x1, x2, sigma=4):
        diff = x1 - x2
        return np.exp(-(np.dot(diff,diff))/ (2 * sigma))

    def k_mat(self,k, X, Y):
        '''
        allowing for N x M kernel for convience in GP problem
        :param k:
        :param X:
        :param Y:
        :return:
        '''

        N = X.shape[0]
        M = Y.shape[0]
        K = np.zeros((N, M))
        for i in range(0, N):
            for j in range(0, M):
                v = k(X[i], Y[j])
                K[i, j] = v
        return np.matrix(K)

    def train(self,X,Y):

        self.X = X.copy()
        N = self.X.shape[0]
        if self.k_func is None:
            self.k_func = self.k_gauss

        self.K_x_x = self.k_mat(self.k_func,X,X)

        self.noise = np.eye(N) * self.sigma

        f = np.random.multivariate_normal(np.zeros((N,)),self.K_x_x,1) # sample function
        #self.Y = np.random.multivariate_normal(f.reshape((N,)),self.noise) # sample observations
        #self.Y = np.matrix(self.Y.reshape((N,1)))
        self.Y = np.matrix(Y.reshape((N,1)))

        self.K_x_x_inv = np.linalg.inv(self.K_x_x + self.noise)

    def plot(self):

        mean = np.array(self.mean)
        cov = np.array(self.cov)

        upper = mean + 2 * np.diag(cov).reshape((mean.shape[0], 1))
        lower = mean - 2 * np.diag(cov).reshape((mean.shape[0], 1))

        #plt.scatter(X, np.reshape(np.array(y), (N,)))

        #plt.plot(X_star, mean, color='red')
        #plt.plot(X_star, upper, color='blue')
        #plt.plot(X_star, lower, color='blue')
        #plt.show()



    def predict(self,X_star,Y_actual):

        self.K_s_s = self.k_mat(self.k_func,X_star,X_star)

        self.K_s_x = self.k_mat(self.k_func,X_star,self.X)

        self.mean = self.K_s_x*self.K_x_x_inv*self.Y
        self.cov = self.K_s_s - self.K_s_x*self.K_x_x_inv*self.K_s_x.T








