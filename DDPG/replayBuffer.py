import numpy as np

class replayBuffer:

    def __init__(self, mem_size, state_space_dim, actions_space_dim):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_space_dim))
        self.new_state_memory = np.zeros((self.mem_size, state_space_dim))
        self.action_memory = np.zeros((self.mem_size, actions_space_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, new_state, reward, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        if batch_size > max_mem:
        	batch_size = max_mem

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_state = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, new_state, rewards, dones