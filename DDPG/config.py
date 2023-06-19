'''
Checkpoint directory per il salvataggio dei modelli Actor e Critic
'''
import os
CHKPT_DIR = 'tmp/ddpg'
if not os.path.isdir(os.path.join(os.getcwd(), CHKPT_DIR)):
    os.mkdir(os.path.join(os.getcwd(), CHKPT_DIR))

'''
Dimensione del replayBuffer
'''
MEM_SIZE = 100000

'''
Dimensione del batch di prelievo dalla memoria e di aggiornamento network
'''
BATCH_SIZE = 64

'''
Numero di step prima dei quali l'agente non effettuerà training. 
Serve a riempire l'experience replay buffer di un numero sufficiente di transizioni prima di iniziare a prelevarle.
'''
DELAY_STEPS = 1024

'''
Numero di step che devono intercorrere tra ogni training dell'agente.
Qualsiasi sia il valore impostato si tenga a mente che il rapporto tra step dell'environment e aggiornamento delle reti rimane fisso ad 1
'''
UPDATE_EVERY = 1

'''
Il dizionario contentente le azioni desiderate e le rispettive funzioni di attivazione all'uscita dell'Actor network
e il (min,max) dell'azione.
Per modificare le azioni che si vuole includere nello schema di controllo agire solo su ACTIONS_ACTFUNC_DICT
'''
ACTIONS_ACTFUNC_DICT = {'accel': ['sigmoid', (-0.9, 0.9)]}
ACTIONS_SPACE = [*ACTIONS_ACTFUNC_DICT.keys()]
ACTIONS_SPACE_DIM = len(ACTIONS_SPACE)
ACTIONS_MIN = [key_elem[1][1][0] for key_elem in ACTIONS_ACTFUNC_DICT.items()]
ACTIONS_MAX = [key_elem[1][1][1] for key_elem in ACTIONS_ACTFUNC_DICT.items()]

'''
Dizionario dei parametri di stato, la chiave indica il nome del parametro è una lista che contiene rispettivamente la dimensione
di quello stato e il valore rispetto al quale normalizzarlo prima di inserirlo nella rete neurale. 
Per aggiungere parametri allo spazio di stato basta aggiungere il loro identificativo, la dimensione e il valore con cui normalizzarli in questo dict
'''
PI= 3.14159265359
STATE_SPACE_DICT = {'xl': [0, 10000], 'vl': [0, 380.], 'xc': [0, 10000], 'vc': [0, 380.]}
STATE_SPACE = [*STATE_SPACE_DICT.keys()]
STATE_SPACE_DIM = len(STATE_SPACE)
#STATE_SPACE_DIM = sum([elem[0] for elem in STATE_SPACE_DICT.values()])
STATE_SPACE_NORM = [v[1] for v in STATE_SPACE_DICT.values() for i in range(v[0])]

'''
Discount factor Q-Learning
'''
DC_FACT = 0.99

'''
Smooth factor per la soft copy tra modelli e modelli target
'''
SM_FACT = 0.001

'''
Learning rate del Critic Network e dell'Actor Network
'''
CLR = 0.001
ALR = 0.001

'''
Numero di neuroni del primo e del secondo livello del Critic Network
'''
CRITIC_1_DIM = 200
CRITIC_2_DIM = 100

'''
Numero di neuroni del primo e del secondo livello dell'Actor Network
'''
ACTOR_1_DIM = 200
ACTOR_2_DIM = 100

'''
Funzione di noise da applicare all'actor per la exploration degli stati
'''
from OU import *
NOISE = OU


