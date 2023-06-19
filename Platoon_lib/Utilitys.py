import matplotlib.pyplot as plt
import numpy as np


def deKron(X0,N,nVars):
    xv = np.zeros(N)
    xp = np.zeros(N)

    for i in range(0, N * nVars, nVars):
        xp[int(i/nVars)] = X0[i]
        xv[int(i/nVars)] = X0[i+1]

    return xp, xv

def Make_Topology(N):
    A = np.zeros((N,N))
    #PF Topology
    for i in range(N):
        A[i, i] = 1                             #diagonal
        if i>=0 and i-1>=0:
            A[i,i-1] = 1                        #next
            if i+1<N:
                A[i,i+1] = 1                    #successor

    L_v = np.dot(A,np.ones((np.size(A,0),1)))
    L = np.zeros((N,N))

    for i in range(N):
        L[i,i] = L_v[i]

    L = L-A

    return A, L

def AB_elem_prod(A,B):

    if (np.size(A,0), np.size(A, 1)) != (np.size(B,0), np.size(B, 1)):
        print('Matrix dimensions dismatch')
        return 0

    R = np.zeros((np.size(A,0), np.size(A, 1)))

    for i in range(np.size(A,0)):
        for j in range(np.size(A, 1)):
            R[i, j] = A[i, j]*B[i, j]

    return R


def dynamic_smoothing(x, box_dim):
    box = np.ones(box_dim)/box_dim
    y = np.convolve(x,box, mode = 'same')
    return y

def inputDesired(mode,v0, T_max = 100, dt = 0.0001):

    if mode == 'trap':
        t_max = T_max
        t = np.linspace(0, t_max, int(t_max / dt))

        r = np.zeros(len(t))
        r[: int(len(t) / 8)] = v0
        r[int(len(t) / 8): int(len(t) * 2 / 8)] = 2 * t[: int(len(t) / 8)] + r[int(len(t) / 8) - 1]
        r[int(len(t) * 2 / 8): int(3 / 8 * len(t))] = r[int(len(t) * 2 / 8) - 1]
        r[int(3 / 8 * len(t)) - 1: int(4 / 8 * len(t))] = r[int(len(t) * 3 / 8) - 5] - 2 * t[: int(len(t) / 8) + 1]
        r[int(4 / 8 * len(t)):] = r[ int(len(t) * 4 / 8)-1]

    elif mode == 'constant':
        t_max = T_max
        t = np.linspace(0, t_max, int(t_max / dt))

        r = v0*np.ones(len(t))

    elif mode == 'sawtooth':

        t_ = np.linspace(0, T_max, int(T_max / dt))

        r_ = np.zeros(len(t_))
        r_[: int(len(t_) / 10)] = v0
        r_[int(len(t_) / 10): int(len(t_) * 3 / 10)] =  0.5 * t_[: int(len(t_) * 2 / 10)] + r_[int(len(t_) / 10)-1]
        r_[int(3 / 10 * len(t_)) - 1: int(5 / 10 * len(t_))] = + r_[int(len(t_) * 3 / 10) - 1] #- 0.5 * t[: int(len(t) * 2 / 10) + 1] + r_[int(len(t) * 3 / 10) - 1]
        r_[int(len(t_) * 5 / 10): int(len(t_) * 7 / 10)] = 0.5 * t_[: int(len(t_) * 2 / 10)] + r_[5 * int(len(t_) / 10) - 2]
        r_[int(len(t_) * 7 / 10): int(len(t_) * 9 / 10)] = + r_[7 * int(len(t_) / 10) - 1] #- 0.5 * t[: int(len(t) * 2 / 10)] + r_[7 * int(len(t) / 10) - 1]
        r_[int(9 / 10 * len(t_)):] = r_[9 * int(len(t_) / 10) - 1]

        r = np.zeros(len(t_) - 1)
        print('I\'m starting the filtering of the desired input')
        r = dynamic_smoothing(r_,len(r_)//10)
        print('DONE')
        t = np.linspace(0, len(r) * (dt), int(len(r)))

    elif mode == 'sin':
        A = 1.5
        t = np.linspace(0,T_max, int(T_max/dt))
        f = 1/50
        r = A * np.sin(2*np.pi*f*t) + v0


    return r, t


def Plotting_error(t, xp, xv, epos, evel, nonLinearDynamics = 0):

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(t, xp, label=['Pos_1', 'Pos_2', 'Pos_3', 'Pos_4', 'Pos_5'])
    plt.title('Position [m]')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(t, xv, label=['Speed_1', 'Speed_2', 'Speed_3', 'Speed_4', 'Speed_5'])
    plt.title('Speed [m/s^2]')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(t, epos, label=['ErrorP_12', 'ErrorP_23', 'ErrorP_34', 'ErrorP_45'])
    plt.title('Errore di posizione')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(t, evel, label=['ErrorV_12', 'ErrorV_23', 'ErrorV_34', 'ErrorV_45'])
    plt.title('Errore di velocità')
    plt.legend()


    if not isinstance(nonLinearDynamics, int):
        plt.figure(2)
        plt.plot(t[:-1], nonLinearDynamics, label=['NonLinear_1', 'NonLinear_2', 'NonLinear_3', 'NonLinear_4'])
        plt.title('NonLinear Dynamics')
        plt.legend()

    plt.show()

