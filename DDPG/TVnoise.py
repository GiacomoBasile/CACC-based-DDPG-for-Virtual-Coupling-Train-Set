from numpy.core.numeric import zeros_like
from numpy.random import normal
from numpy import array

'''
Questo rumore è gaussiano con media MU_1 e std STD_1 per le prime 50mila volte che viene chiamato e poi 
tende linearmente nei successivi 100mila passi ad uno con media MU_2 e std STD_2.
Nei successivi 150mila passi decade linearmente a zero
'''

#media e std del rumore gaussiano per i primi 50mila step
MU_1 = array([[0.30 , 0.0 , 0.0]], dtype = 'float32')
STD_1 = array([[0.20 , 0.05 , 0.30]], dtype = 'float32')

#media e std del rumore gaussiano target ai 150mila passi
MU_2 = array([[0.60 , -0.1 , 0.0]], dtype = 'float32')
STD_2 = array([[0.10 , 0.05 , 0.30]], dtype = 'float32')

#media e std del rumore target a 300000 passi
MU_3 = array([[0.0 , 0.0 , 0.0]], dtype = 'float32')
STD_3 = array([[0.0 , 0.0 , 0.0]], dtype = 'float32')

INIT_COUNTR = 266432

class TVNoise:

    def __init__(self, mu = [MU_1,MU_2,MU_3], std = [STD_1,STD_2,STD_3], init_countr = INIT_COUNTR):
        
        self.mu = mu
        self.std = std
        self.time_step1 = 50000
        self.time_step2 = 150000
        self.time_step3 = 300000
        if init_countr is None:
            self.countr = -1
        else: 
            self.countr = init_countr - 1

    def __call__(self):

        self.countr += 1

        if self.countr < self.time_step1:
            return self.mu[0] + self.std[0] * normal(size = self.mu[0].shape)
        
        elif self.countr >= self.time_step1 and self.countr < self.time_step2:
            
            mu = (self.mu[0] - self.mu[1])/(self.time_step1 - self.time_step2) * self.countr + (self.time_step1*self.mu[1] - self.time_step2*self.mu[0])/(self.time_step1 - self.time_step2)
            std = (self.std[0] - self.std[1])/(self.time_step1 - self.time_step2) * self.countr + (self.time_step1*self.std[1] - self.time_step2*self.std[0])/(self.time_step1 - self.time_step2)
            
            return mu + std * normal(size = mu.shape)
        
        elif self.countr >= self.time_step2 and self.countr < self.time_step3:
            
            mu = (self.mu[1] - self.mu[2])/(self.time_step2 - self.time_step3) * self.countr + (self.time_step2*self.mu[2] - self.time_step3*self.mu[1])/(self.time_step2 - self.time_step3)
            std = (self.std[1] - self.std[2])/(self.time_step2 - self.time_step3) * self.countr + (self.time_step2*self.std[2] - self.time_step3*self.std[1])/(self.time_step2 - self.time_step3)
            
            return mu + std * normal(size = mu.shape)
        
        else:
           return zeros_like(self.mu[0])