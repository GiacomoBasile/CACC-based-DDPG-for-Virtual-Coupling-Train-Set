import sys
sys.path.append('./Platoon_lib/')
sys.path.append('./DDPG/')
from Platoon_lib import Model_Impl

import matplotlib.pyplot as plt
import numpy as np
import virtualcopling as vc
from DDPG import agent

from utils import plot_learning_curve
from Platoon_lib.Utilitys import inputDesired




if __name__=='__main__':

    N = 2
    ddes = 5
    Nvars = 2

    datiVeicolo_T = {'m': 475000,
                     'c1': 0.8420,
                     'c2': 0.00681,
                     'c3': 0.00209,
                     'w': 0.1}


    mode = 'constant'  # Mode: sawtooth, trap, constant
    scale = 60.0  # V0_leader
    T_max = 5
    dt = 0.01

    reference, t = inputDesired(mode, scale, T_max, dt)
    x0 = Model_Impl.getInitConditions(N, scale, Nvars)

    env = vc.VirtualCoplingEnv(datiVeicolo=datiVeicolo_T, x0=x0, t=t, type='Train')
    agent = agent.Agent(training=True)

    episode_NL = 0
    n_games = 10000
    figure_file = './plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = True


    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            r = reference[0]
            observation = env.reset()
            act = env.action_space.sample()
            actions = np.array([r, act], dtype=np.float32)
            observation_, reward, done, info, input = env.step(actions)
            agent.remember(state=observation, action=act, reward=reward, new_state=observation_, done=done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
        load_checkpoint = False
    else:
        evaluate = False

    im_learning = False
    for i in range(n_games):

        observation = env.reset()
        done = False
        score = 0
        idx_r = 0
        while not done and idx_r < T_max/dt:
            r = reference[idx_r]
            act, n = agent.choose_action(state=observation)
            actions = np.array([r, act], dtype=np.float32)
            observation_, reward, done, info, u = env.step(actions)
            score += reward
            agent.remember(state=observation, action=act, reward=reward, new_state=observation_, done=done)

            if not load_checkpoint and i >= episode_NL:
                im_learning = True
                agent.learn()
            observation = observation_
            idx_r += 1
            print('Episode ', i, '|Time: %.2f' % (idx_r*dt), '| learning: ', im_learning,
                  '|Error: %.2f' % (observation[0]-observation[2] -ddes), '|Done:', done,
                  '|Observations:%.2f,' %observation[0], ' %.2f,' %observation[1], '%.2f' %observation[2], '%.2f' %observation[3],
                  '| Input: %.2f,' %r, ' %.2f,' %act, '%.2f' %u,' %.2f' %n)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        #if avg_score > best_score:
        #    best_score = avg_score
        #    if not load_checkpoint:
        #        agent.save_models(count=i)
        if not load_checkpoint:
            agent.save_models(count=i)
        print('################################################################################################################################################################################################')
        print('Episode ', i, '| Score: %.1f' % score, '|Avg score: %.1f' % avg_score, '| Info:', info)
        print('################################################################################################################################################################################################')

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)