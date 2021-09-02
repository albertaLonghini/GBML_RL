import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from networks import DQN
"""
Class which implements the memory buffer. It stores the episodes as a circular memory, which means that once the memory
is full it starts substituting elements from the oldest one. In this way the memory keeps being up to date with the 
actions chosen from the trained network.
The episodes are stored as a dictionary:
element = {'st': st, 'a': a, 'r': r, 'terminal': terminal, 'st1': st1}

It has a function that randomly samples a batch of episodes from its memory.
"""
class Memory(object):

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, st, a, r, terminal, st1):
        if len(self.memory) < self.size:
            self.memory.append(None)

        element = {'st': st, 'a': a, 'r': r, 'terminal': terminal, 'st1': st1}

        self.memory[int(self.position)] = element
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




"""
Class that implements the Q value functions. It has two identical network, one that is actually trained and one
that is updated once every self.update_target_episodes with the weights of the other network in order to make
the training more stable. Other fucntions are:
- get_action
- get_tensor
- push_memory
- update_Q
- update_target
- write_reward
- get_Q_grid
"""
class Net(nn.Module):

    def __init__(self, params, logdir, device):
        super(Net, self).__init__()

        memory_size = 100000
        lr = 0.001
        self.min_memory = 1000
        self.update_target_episodes = 100
        self.batch_size = 128
        self.gamma = 0.9  # 0.97
        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_decay = 0.97 #0.997
        log_dir = logdir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_idx = 0
        self.device = device

        self.Q_model = DQN(params['grid_size'], params['model_type'])

        # Target network updated only once every self.update_target_episodes
        self.Q_target = DQN(params['grid_size'], params['model_type'])
        self.Q_target.model.load_state_dict(self.Q_model.model.state_dict())
        self.Q_target.eval()

        self.D = Memory(memory_size)
        self.optimizer = optim.RMSprop(self.Q_model.parameters(), lr=lr)

    def get_action(self, x, test=False):
        """
        This function return the action based on the state (the grid with the position and the goal) and it gets it
        either from the network or from the exploration algorithm (epsilon-greedy right now)

        :param x: actual state of the grid
        :type x: nparray
        :param test: true if we are in a test fase where we don't need the exploration
        :type test: bool
        :return a: action that the agent has to take
        :rtype: int
        """
        if random.random() > self.epsilon or test:
            q = torch.softmax(self.Q_model(self.get_tensor(x)), -1)
            a = torch.argmax(q).detach().cpu().item()
        else:
            a = np.random.randint(0, 4)
        return a

    def get_tensor(self, state):
        return torch.from_numpy(state).float().to(self.device)

    def push_memory(self, s, a, r, t, s1):
        self.D.push(s, a, r, t, s1)

    def update_Q(self):
        """
        This function updates the Q_value network by sampling a batch of episodes form the buffer
        {st, a, r, terminal, st1}
        and using them to compute the TD error for the specific action "a":

        r + gamma * argmax(Q_target(st1)) - Q_value(st)    if not terminal state
        r - Q_value(st)                                    if terminal state

        """
        if len(self.D) < self.min_memory:
            return

        self.Q_model.train()

        # sample action from the buffer and store in separate elements state, action taken, reward received and following state
        data = self.D.sample(self.batch_size)

        st = torch.cat([self.get_tensor(x['st']) for x in data], 0)

        a = [x['a'] for x in data]
        r = torch.cat([torch.tensor(x['r'], dtype=torch.float32).view(1) for x in data], 0).to(self.device)
        terminal = torch.cat([torch.tensor(x['terminal'] * 1.0, dtype=torch.float32).view(1) for x in data], 0).to(self.device)
        st1 = torch.cat([self.get_tensor(x['st1']) for x in data], 0)

        # Compute value of st from target network by r + gamma* argmax(Q_target(st1))
        Qt1 = self.Q_target(st1)
        max_vals, _ = torch.max(Qt1, -1)
        y = (r + terminal * (self.gamma * max_vals)).detach()

        # Compute value of st from Q_value network by Q(st) and get the Q value just for the action given from the buffer
        Q = self.Q_model(st)
        Q = torch.cat([Q[i, a[i]].view(1) for i in range(len(a))], 0)

        # Compute the loss that corresponds to the Temporal Difference error
        TDerror = (y - Q) ** 2
        loss_q = torch.mean(TDerror)
        loss = loss_q

        # backprop from the mean of the TD losses in the batch
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return

    def update_target(self, episode, grid=None):
        """
        Update the target network (self.Q_target) every self.update_target_episodes and decays epsilon for
        the exploration.
        Moreover it calls the get_Q_grid function to generate a table that shows the Q-values for each possible
        state-action combination.

        :param episode: actual episode
        :type episode: int
        :param grid: actual state
        :type grid: nparray
        """
        if episode % self.update_target_episodes == 0:
            if grid != None:
                _ = self.get_Q_grid(grid)
            self.Q_target.model.load_state_dict(self.Q_model.model.state_dict())
            self.Q_target.eval()
            self.epsilon *= self.epsilon_decay

    def write_reward(self, r, r2):
        """
        Function that write on tensorboard the rewards it gets

        :param r: cumulative reward of the episode
        :type r: float
        :param r2: final reword of the episode
        :type r2: float
        """

        self.writer.add_scalar('cumulative_reward', r, self.log_idx)
        self.writer.add_scalar('final_reward', r2, self.log_idx)
        self.log_idx += 1
