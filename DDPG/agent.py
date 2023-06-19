from config import *
from networks import *
from replayBuffer import *
import tensorflow as tf
import numpy as np
from OU import *

class Agent:

	def __init__(self, state_space_dim=STATE_SPACE_DIM , actions_actfunc_dict=ACTIONS_ACTFUNC_DICT,
				 actions_space_dim=ACTIONS_SPACE_DIM, clr = CLR, alr = ALR, dc_fact = DC_FACT,
				 sm_fact = SM_FACT, actor_1_dim = ACTOR_1_DIM, actor_2_dim = ACTOR_2_DIM,
				 critic_1_dim = CRITIC_1_DIM, critic_2_dim = CRITIC_2_DIM,
				 mem_size = MEM_SIZE, batch_size = BATCH_SIZE,
				 delay_steps = DELAY_STEPS, update_every = UPDATE_EVERY,
				 noise = NOISE, training = False, chkpt_dir = CHKPT_DIR):

		self.critic = Critic(state_space_dim = state_space_dim, actions_space_dim=actions_space_dim, clr=clr, critic_1_dim=critic_1_dim,
							 critic_2_dim=critic_2_dim, name='Critic', chkpt_dir=chkpt_dir)

		self.target_critic = Critic(state_space_dim = state_space_dim, actions_space_dim=actions_space_dim, clr=clr, critic_1_dim=critic_1_dim,
									critic_2_dim=critic_2_dim, name='Target_Critic', chkpt_dir=chkpt_dir)

		self.actor = Actor(state_space_dim = state_space_dim, actions_actfunc_dict=actions_actfunc_dict, alr=alr, actor_1_dim=actor_1_dim,
						   actor_2_dim=actor_2_dim, name='Actor', chkpt_dir=chkpt_dir)

		self.target_actor = Actor(state_space_dim = state_space_dim, actions_actfunc_dict=actions_actfunc_dict, alr=alr, actor_1_dim=actor_1_dim,
								  actor_2_dim=actor_2_dim, name='Target_Actor', chkpt_dir=chkpt_dir)

		self.replayBuffer = replayBuffer(mem_size=mem_size, state_space_dim=state_space_dim, actions_space_dim=actions_space_dim)

		self.dc_fact = dc_fact
		self.sm_fact = sm_fact
		self.batch_size = batch_size
		self.noise = noise()
		self.training = training
		self.delay_steps = delay_steps
		self.update_every = update_every
		self.global_steps = 0

		self._update_target(sm_fact=1)

	def save_models(self, count):
		self.critic.save_model(count)
		self.target_critic.save_model(count)
		self.actor.save_model(count)
		self.target_actor.save_model(count)

	def load_models(self, count = None):

		if count is None:
			episodes = [int(elem) for elem in os.listdir(os.path.join(os.getcwd(), CHKPT_DIR)) if os.path.isdir(os.path.join(os.getcwd(), CHKPT_DIR, elem))]
			if len(episodes) == 0:
				return -1
			
			episodes.sort()
			count = episodes[-1]
		
		try:
			self.critic.load_model(count)
			self.target_critic.load_model(count)
			self.actor.load_model(count)
			self.target_actor.load_model(count)
		except:
			return -2

		return 0

	def choose_action(self, state):

		state = np.array(state, dtype = 'float32')
		state = state.reshape(1,-1)
		actions = self.actor(state)
		action = actions[0][0]

		if self.training and self.noise is not None:
			n = self.noise()
			action = action + n
			action = action.clip(ACTIONS_MIN, ACTIONS_MAX)

		return action,n
		

	def learn(self):
		
		self.global_steps += 1

		if self.replayBuffer.mem_cntr < self.batch_size or self.training == False or self.global_steps < self.delay_steps or self.global_steps % self.update_every != 0:
			return
		
		for _ in range(self.update_every):
			states, actions, new_states, rewards, dones = self.replayBuffer.sample_buffer(self.batch_size)
			rewards = rewards.reshape(-1,1)
			dones = dones.reshape(-1,1)
			states = tf.convert_to_tensor(states, dtype=tf.float32)
			actions = tf.convert_to_tensor(actions, dtype=tf.float32)
			new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
			rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

			with tf.GradientTape() as tape:
				target_actions = self.target_actor.model(new_states, training = True)
				new_critic_value = self.target_critic.model([new_states, target_actions], training = True)
				critic_value = self.critic.model([states, actions], training = True)
				target = rewards + self.dc_fact * new_critic_value * (1-dones)
				critic_loss = tf.keras.losses.MSE(target, critic_value)

			critic_network_gradient = tape.gradient(critic_loss, self.critic.model.trainable_variables)
			self.critic.opt.apply_gradients(zip(critic_network_gradient, self.critic.model.trainable_variables))
		
			with tf.GradientTape() as tape:
				new_policy_actions = self.actor.model(states, training = True)
				critic_value = -self.critic.model([states, new_policy_actions], training=True)
				actor_loss = tf.math.reduce_mean(critic_value)
			
			actor_network_gradient = tape.gradient(actor_loss, self.actor.model.trainable_variables)
			self.actor.opt.apply_gradients(zip(actor_network_gradient, self.actor.model.trainable_variables))
			
			self._update_target()

	def remember(self, state, action, new_state, reward, done = False):
		self.replayBuffer.store_transition(state, action, new_state, reward, done)

	def _update_target(self, sm_fact = None):

		if sm_fact is None:
			sm_fact = self.sm_fact

		target_critic_weights = self.target_critic.model.get_weights()
		new_target_critic_weights = []

		for i, elem in enumerate(self.critic.model.get_weights()):
			new_target_critic_weights.append( sm_fact * elem + (1 - sm_fact) * target_critic_weights[i] )

		self.target_critic.model.set_weights(new_target_critic_weights)


		target_actor_weights = self.target_actor.model.get_weights()
		new_actor_critic_weights = []

		for i, elem in enumerate(self.actor.model.get_weights()):
			new_actor_critic_weights.append( sm_fact * elem + (1 - sm_fact) * target_actor_weights[i] )

		self.target_actor.model.set_weights(new_actor_critic_weights)