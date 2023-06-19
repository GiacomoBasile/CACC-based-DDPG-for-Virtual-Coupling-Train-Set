import numpy as np
from random import random

#valori di theta, media e std del processo
THETA_1 = .2
MU_1 = 0.0
SIGMA_1 = .3

#valori necessari nel caso dello stochastic brake
#https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
THETA_2 = -.5
MU_2 = -.5
SIGMA_2 = .1

class OU:
    def __init__(self, mu = [MU_1,MU_2], sigma = [SIGMA_1,SIGMA_2], theta=[THETA_1,THETA_2], dt=1e-3, x_initial = None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x_initial = x_initial
        self.dt = dt
        self.reset()

    def __call__(self):
        
        if random() <= 0.1:

            x = self.mu[1] + self.sigma[1] * np.random.normal(size=1)
        
        else:

            x = (
                self.x_prev
                + self.theta[0] * (self.mu[0] - self.x_prev) * self.dt
                + self.sigma[0] * np.sqrt(self.dt) * np.random.normal(size=1)
            )
            self.x_prev = x

        # Store x into x_prev
        # Makes next noise dependent on current one

        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mu[0])