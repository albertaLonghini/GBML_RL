import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from RL_Project.networks import DQN_Net
# from networks import DQN_Net


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


class DQN(nn.Module):

    def __init__(self, params, logdir, device):
        super(DQN, self).__init__()

        memory_size = 100000
        lr = 0.001
        self.min_memory = 1000
        self.update_target_episodes = 100
        self.batch_size = 128
        self.gamma = 0.9  # 0.97
        self.epsilon0 = 1.0 #0.9
        self.epsilon = self.epsilon0
        self.epsilon_decay = 0.98 #0.99 #0.97 #0.97 #0.997
        log_dir = logdir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_idx = 0
        self.device = device

        self.Q_model = DQN_Net(10)

        # Target network updated only once every self.update_target_episodes
        self.Q_target = DQN_Net(10)
        self.Q_target.model.load_state_dict(self.Q_model.model.state_dict())
        self.Q_target.eval()

        self.D = Memory(memory_size)
        self.optimizer = optim.RMSprop(self.Q_model.parameters(), lr=lr)

    def get_action(self, x, test=False):

        if random.random() > self.epsilon or test:
            q = torch.softmax(self.Q_model(self.get_tensor(x)), -1)
            a = torch.argmax(q).detach().cpu().item()
        else:
            a = np.random.randint(0, 4)
        return a, None

    def get_tensor(self, state):
        return torch.from_numpy(state).float().to(self.device)

    def push_data(self, s, a, logprob, r, done, s1):
        self.D.push(s, a, r, (not done), s1)

    def update(self, epoch):
        self.update_Q()
        self.update_target(epoch)

    def update_Q(self):

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

    def update_target(self, episode):

        if episode % self.update_target_episodes == 0:
            self.Q_target.model.load_state_dict(self.Q_model.model.state_dict())
            self.Q_target.eval()
            self.epsilon *= self.epsilon_decay

    def write_reward(self, r, r2):

        self.writer.add_scalar('Test cumulative reward current mazes', r, self.log_idx)
        self.writer.add_scalar('Test final reward current mazes', r2, self.log_idx)
        self.log_idx += 1
