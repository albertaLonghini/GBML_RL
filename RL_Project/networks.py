import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


"""
General One-Head Architecture
"""
class Single_Head(nn.Module):

    def __init__(self, grid_size):
        super(Single_Head, self).__init__()
        img_reduced_dim = grid_size + 2 - 2 * 4

        self.model = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=1),
            nn.Tanh(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.Tanh(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.Tanh(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 4, bias=True)
        )

    def forward(self, x):
        return self.model(x)

"""
General Two-Head Architecture
"""
class Two_Head(nn.Module):

    def __init__(self, grid_size):
        super(Two_Head, self).__init__()
        img_reduced_dim = grid_size + 2 - 2 * 4

        self.body = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=1),
            nn.Tanh(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.Tanh(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.Tanh(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.Tanh(),
            nn.Flatten(),
        )

        self.actor = nn.Sequential(
            nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 4, bias=True)
        )

        self.critic = nn.Sequential(
            nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 1, bias=True)
        )

    def forward_pi(self, x):
        return self.actor(self.body(x))

    def forward_V(self, x):
        return self.critic(self.body(x))

    def forward(self, x):
        return self.actor(self.body(x)), self.critic(self.body(x))


"""
DQN network
"""
class DQN_Net(nn.Module):

    def __init__(self, grid_size):
        super(DQN_Net, self).__init__()

        self.model = Single_Head(grid_size)

    def forward(self, x):
        return self.model(x)


"""
REINFORCE network
"""
class Actor(nn.Module):

    def __init__(self, grid_size):
        super(Actor, self).__init__()

        self.model = Single_Head(grid_size)
        self.softmax = nn.Softmax(dim=-1)

    def get_action(self, state):
        action_probs = self.softmax(self.model(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob

    def evaluate(self, state, action):
        action_probs = self.softmax(self.model(state))
        action_logprobs = torch.log(torch.max(action_probs[0, action], torch.tensor(1e-5)))

        return action_logprobs


"""
PPO network
"""
class ActorCritic(nn.Module):

    def __init__(self, grid_size):
        super(ActorCritic, self).__init__()

        self.model = Two_Head(grid_size)
        self.softmax = nn.Softmax(dim=-1)

    def get_action(self, state):
        action_probs = self.softmax(self.model.forward_pi(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob

    def evaluate(self, state, action):

        pi, state_value = self.model.forward(state)

        action_probs = self.softmax(pi)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


'''
MAML-PPO network
'''
class MamlParamsPPO(nn.Module):

    def __init__(self, grid_size, lr, adaptive_lr=False):
        super(MamlParamsPPO, self).__init__()

        self.img_reduced_dim = grid_size + 2 - 2 * 4
        self.activation_f = nn.Tanh()

        self.theta_shapes = [[4, 3, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                             [4, 32], [4],
                             [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                             [1, 32], [1],
                             ]

        if adaptive_lr:
            self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.theta_shapes))
        else:
            self.lr = [lr] * len(self.theta_shapes)

        self.theta_0 = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
        for i in range(len(self.theta_0)):
            if self.theta_0[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta_0[i])

    def get_theta(self):
        return self.theta_0

    def get_size(self):
        return np.sum([np.prod(x) for x in self.theta_shapes])

    def f_theta(self, x, theta=None):

        if theta is None:
            theta = self.theta_0

        h = self.activation_f(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
        h = h.contiguous()
        h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        h_a = self.activation_f(F.linear(h, theta[8], bias=theta[9]))
        h_a = F.linear(h_a, theta[10], bias=theta[11])
        pi = F.softmax(h_a, -1)
        h_c = self.activation_f(F.linear(h, theta[12], bias=theta[13]))
        h_c = F.linear(h_c, theta[14], bias=theta[15])
        v = h_c[:, 0]

        return pi, v


    def forward(self, x, theta=None):

        pi, v = self.f_theta(x, theta=theta)

        dist = Categorical(pi)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob, v

    def evaluate(self, x, action, theta=None):

        pi, v = self.f_theta(x, theta=theta)

        if torch.is_tensor(action):
            action = action.tolist()
        else:
            action = [action]

        action_logprobs = torch.log(torch.clamp(pi[range(len(action)), action], min=1e-10))

        dist = Categorical(pi)
        dist_entropy = dist.entropy()

        return action_logprobs, v, dist_entropy












