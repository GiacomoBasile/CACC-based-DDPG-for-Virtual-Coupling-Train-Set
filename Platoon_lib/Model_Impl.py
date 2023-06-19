from Utilitys import *
import gym
import numpy as np
from gym import spaces

def  getInitConditions(N,v0,nVars):          # output [x0,x0Adapt,x0Vel]

    x0Principale = [80,v0,75,v0,62,12,58,8,50,5,45,8,38,4,33,5,29,2,26,3,20,10,17,15,11,12,8,4,5,0,3,8,2,3,0,5,0,4,0,0,np.zeros((1,2*N*N))];


    return x0Principale[0:N*nVars]


class Vehicle():
    def __init__(self,x0, datiVeicolo, t, leader=False, type='Car'):

        self.x0 = x0
        self.x1 = [self.x0[0]]
        self.x2 = [self.x0[1]]

        if not leader and type=='Car':
            self.M = datiVeicolo.get('m')
            self.Ca = datiVeicolo.get('Ca')
            self.g = datiVeicolo.get('g')
            self.f = datiVeicolo.get('f')
            self.eta = datiVeicolo.get('eta')
            self.R = datiVeicolo.get('R')

        elif not leader and type=='Train':
            self.M = datiVeicolo.get('m')
            self.C3 = datiVeicolo.get('c3')
            self.C2 = datiVeicolo.get('c2')
            self.C1 = datiVeicolo.get('c1')
            self.alpha = datiVeicolo.get('alpha')
            self.w = datiVeicolo.get('w')

        self.dt = t[1]-t[0]
        self.leader = leader
        self.type = type

    def rinizialaze_states(self):
        del self.x1
        del self.x2

        self.x1 = [self.x0[0]]
        self.x2 = [self.x0[1]]

    def OdeSolver(self, u):
        x_dot = np.zeros(2)
        if not self.leader:
            dynamic = (1 / self.M) * (self.Ca * self.x2[-1]**2 + self.g * self.f)
            x_dot[0] = self.x2[-1]
            x_dot[1] = dynamic + u
        else:
            x_dot[0] = self.x2[-1]
            x_dot[1] = u

        self.x1.append(x_dot[0])
        self.x2.append(x_dot[1])
        return x_dot

    def EuleranSolver(self, u):

        if self.type=='Car':
            if not self.leader:
                dynamic = (1 / self.M) * (self.Ca * self.x2[-1]**2 + self.g * self.f)
                self.x1.append(self.x1[-1] + (self.x2[-1] * self.dt))
                self.x2.append(self.x2[-1] + (dynamic + u) * self.dt)
            else:
                self.x1.append(self.x1[-1] + (self.x2[-1] * self.dt))
                self.x2.append(u)
        elif self.type=='Train':
            if not self.leader:
                dynamic = -(1 / self.M) * (np.sign(self.x2[-1])) * (self.C3 * self.x2[-1]**2 + self.C2 * np.abs(self.x2[-1]) + self.C1)
                self.x1.append(self.x1[-1] + (self.x2[-1] * self.dt))
                self.x2.append(self.x2[-1] + (dynamic + u - self.w) * self.dt)
            else:
                self.x1.append(self.x1[-1] + (self.x2[-1] * self.dt))
                self.x2.append(u)


    def get_obs(self):
        return self.x1[-1], self.x2[-1]

    def get_status(self):
        return self.x1, self.x2

    def get_pos(self):
        return self.x1

    def get_speed(self):
        return self.x2


"""
#   Main test   #
def Main_test():  

  N = 5
  ddes = 5
  Nvars = 2

  datiVeicolo_T = { 'm' : 268056,
                  'c1' : 4615.968,
                  'c2' : 81.365,
                  'c3' : 1.824,
                  'w' : 0.02,
                  'alpha' : 1}

  datiVeicolo_C = { 'm' : 1500,
                  'Ca' : 0.43,
                  'f' : 0.02,
                  'g' : 9.81,
                  'eta' : 0.85,
                  'R' : 0.28}


  # INPUT AND CONTROL DEFINITION
  SAT = 0
  mode = 'constant'  # Mode: sawtooth, trap, constant
  scale = 50.0  # V0_leader
  T_max = 100
  dt = 0.001

  r, t = inputDesired(mode, scale, T_max, dt)

  x0 = getInitConditions(N, scale, Nvars)

  Car = Vehicle(x0=[x0[2], x0[3]], t=t, datiVeicolo=datiVeicolo_T, type='Train')
  leader = Vehicle(x0=[x0[0], x0[1]], t=t, datiVeicolo=datiVeicolo_T, leader=True, type='Train')

    

  for i, t_ in enumerate(t):
    x1, x2 = Car.get_obs()
    xl1, xl2 = leader.get_obs()

    ul = 50
    uc = -(48)*(- xl1 - xl2 + x1 + x2 + 5) #* datiVeicolo.get('eta') / (datiVeicolo.get('m') * datiVeicolo.get('R'))) *
    Car.EuleranSolver(uc)
    leader.EuleranSolver(ul)


  x, v = Car.get_status()
  xl, vl = leader.get_status()

#  return x,v,xl,vl,t
#  print('x', x)
#  print('v', v)
  f1 = plt.figure(1)
  plt.plot(t, x[:-1])
  plt.plot(t, xl[:-1])

  f2 = plt.figure(2)
  plt.plot(t, v[:-1])
  plt.plot(t, vl[:-1])

  return x,v,xl,vl,t
"""