import numpy as np
import matplotlib.pyplot as plt
import os
from DDPG import agent

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_saved_models_trend(path, n_models, model):
    os.chdir(path)
    ddpg_agent = agent
    lst_elem = os.listdir()
    every = int(int(lst_elem[-1]) / n_models)

    for t in range(lst_elem[-1]):
        model.load_models(count=t)
        cont = t*every
    return 0

if __name__=='__main__':
    plot_saved_models_trend(path='./tmp/DDpg_2', every=500)


    print(every)

