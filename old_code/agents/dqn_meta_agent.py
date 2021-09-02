import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from networks import MamlParamsDQN
import numpy as np
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

    def reset(self):
        self.position = 0
        self.memory = []

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

        memory_size = 10000
        lr = 0.001
        self.batch_size = 32
        self.gamma = 0.9  # 0.97
        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_decay = 0.7
        log_dir = logdir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_idx = 0
        self.device = device

        self.inner_lr = 0.1
        self.training_steps = 10

        self.model_theta = MamlParamsDQN(params['grid_size'])

        self.D = Memory(memory_size)
        self.optimizer = optim.RMSprop(self.model_theta.parameters(), lr=lr)

    def get_action(self, x, theta=None, test=False):

        q = torch.softmax(self.model_theta(self.get_tensor(x), theta=theta), -1)
        if random.random() > self.epsilon or test:
            a = torch.argmax(q).detach().cpu().item()
        else:
            a = np.random.randint(0, 4)
        return a

    def get_tensor(self, state):
        return torch.from_numpy(state).float().to(self.device)

    def push_memory(self, s, a, r, t, s1):
        self.D.push(s, a, r, t, s1)

    def adapt(self, s, a, r, done, s1, train=False):

        self.model_theta.train()

        theta_i = self.model_theta.get_theta()

        s = self.get_tensor(s)
        r = torch.tensor(r, dtype=torch.float32).view(1).to(self.device)
        terminal = torch.tensor(done * 1.0, dtype=torch.float32).view(1).to(self.device)
        s1 = self.get_tensor(s1)

        Q1 = self.model_theta(s1)
        max_vals, _ = torch.max(Q1, -1)
        y = (r + terminal * (self.gamma * max_vals)).detach()

        Q = self.model_theta(s)
        Q = Q[0, a].view(1)

        TDerror = (y - Q) ** 2
        loss_q = torch.mean(TDerror)

        if train:
            theta_grad_s = torch.autograd.grad(outputs=loss_q, inputs=theta_i, create_graph=True)
            theta_i = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(theta_grad_s, theta_i)))
        else:
            theta_grad_s = torch.autograd.grad(outputs=loss_q, inputs=theta_i)
            theta_i = list(map(lambda p: p[1] - self.inner_lr * p[0].detach(), zip(theta_grad_s, theta_i)))

        return theta_i


    def update_Q(self, theta):

        self.model_theta.train()

        tot_loss = 0

        for _ in range(self.training_steps):

            # sample action from the buffer and store in separate elements state, action taken, reward received and following state
            data = self.D.sample(self.batch_size)

            st = torch.cat([self.get_tensor(x['st']) for x in data], 0)

            a = [x['a'] for x in data]
            r = torch.cat([torch.tensor(x['r'], dtype=torch.float32).view(1) for x in data], 0).to(self.device)
            terminal = torch.cat([torch.tensor(x['terminal'] * 1.0, dtype=torch.float32).view(1) for x in data], 0).to(self.device)
            st1 = torch.cat([self.get_tensor(x['st1']) for x in data], 0)

            # Compute value of st from target network by r + gamma* argmax(Q_target(st1))
            Qt1 = self.model_theta(st1, theta)
            max_vals, _ = torch.max(Qt1, -1)
            y = (r + terminal * (self.gamma * max_vals)).detach()

            # Compute value of st from Q_value network by Q(st) and get the Q value just for the action given from the buffer
            Q = self.model_theta(st, theta)
            Q = torch.cat([Q[i, a[i]].view(1) for i in range(len(a))], 0)

            # Compute the loss that corresponds to the Temporal Difference error
            TDerror = (y - Q) ** 2
            loss_q = torch.mean(TDerror)
            tot_loss += loss_q

        return tot_loss

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


