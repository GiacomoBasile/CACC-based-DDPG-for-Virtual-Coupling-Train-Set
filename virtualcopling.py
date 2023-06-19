import gym
import numpy as np
from gym import spaces
from os import path
from Platoon_lib.Model_Impl  import Vehicle


class VirtualCoplingEnv(gym.Env):
    def __init__(self, datiVeicolo, x0, t, type='car', high=np.array([100000, 320, 100000, 320], dtype=np.float32)):
        self.N_var = 2
        self.N_vehicle = 2
        self.max_speed = 2.0
        self.max_torque = 0.9
        self.x0 = x0



        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)


        self.leader_car = Vehicle(x0=self.x0[:self.N_var], datiVeicolo=datiVeicolo, t=t, leader=True, type=type)
        self.car = Vehicle(x0=x0[-self.N_var:], datiVeicolo=datiVeicolo, t=t, leader=False, type=type)

    def reset(self):

        self.leader_car.rinizialaze_states()
        self.car.rinizialaze_states()
        xl, vl = self.leader_car.get_obs()
        xc, vc = self.car.get_obs()

        return np.array([xl, vl, xc, vc], dtype=np.float32)

        #return np.array([xl-xc,  vl-vc], dtype=np.float32)


    def step(self, action=None, ddes=5.0):

        done = False
        info = ''
        cond, cond_, cond__ = '', '', ''

        self.leader_car.EuleranSolver(action[0])
        #u = np.clip(action[1], -self.max_torque, self.max_torque)
        u = action[1]
        self.car.EuleranSolver(u)

        xl, vl = self.leader_car.get_obs()
        xc, vc = self.car.get_obs()


        if (xl-xc) <= ddes or vc < 0.0 or (xl-xc) > 600:
            if xl-xc < 0.0 or (xl-xc) > 600:
                done = True
                cond = 'xl-xc < 0 (Accident)' * ((xl - xc) <= 0.0)
                cond__ = 'To far restart' * ((xl - xc) > 600)
            rc = 100

            info = 'Worst error: '
            cond_ = 'Ego car speed < 0' * (vc < 0.0)

        else:
            rc = 0

        ra = (vl - vc)**2
        rd = np.abs(((xl-xc)/ddes) - 1)
        costs = 0.1 * ra + rd + rc

        obs = np.array([xl, vl, xc, vc], dtype=np.float32)

        return obs, -costs, done, info+cond+cond_+cond__, u
"""
    def step(self, action=None, ddes=0):

        done = False
        info = ''
        cond, cond_, cond__ = '', '', ''

        self.leader_car.EuleranSolver(action[0])
        u = np.clip(action[1], -self.max_torque, self.max_torque)
        self.car.EuleranSolver(u)

        xl, vl = self.leader_car.get_obs()
        xc, vc = self.car.get_obs()

        #print('x: ', [xl, xc])
        #print('v: ', [vl, vc])

        if np.abs(xl - xc)<ddes:
            termm = 1000
        else:
            termm = 0

        costs =  (vc/vl) * (xl - xc - ddes)**2 + termm #+ 0.5 * u**2 + (vc/vl) * (vl-vc)**2   +

        if (xl-xc) <= ddes or vc < 0.0 or (xl-xc) > 600:
            done = True
            costs = 10000000

            info = 'Worst error: '
            cond = 'xl-xc < 0 (Accident)' * ((xl-xc) <= ddes)
            cond_ = 'Ego car speed < 0' * (vc < 0.0)
            cond__ = 'To far restart' * ((xl-xc) > 600)
        #    print(cond, (xl-xc))
        #    print(cond_, vc)
        #    print(cond__, (xl-xc))
        #    input('booh')

        obs = np.array([xl, vl, xc, vc], dtype=np.float32)
        #obs = np.array([xl-xc,  vl-vc], dtype=np.float32)
        #input('wait')

        return obs, -costs, done, info+cond+cond_+cond__, u
"""