import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


"""
DQN network
"""


class DQN(nn.Module):

    def __init__(self, grid_size, model=0):
        super(DQN, self).__init__()

        if model == 0:

            img_reduced_dim = grid_size + 2 - 2*4

            self.model = nn.Sequential(
                nn.Conv2d(3, 4, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
                nn.ReLU(),
                nn.Linear(32, 4, bias=True)
            )

        elif model == 1:

            img_reduced_dim = grid_size + 2 - 2 * 6

            self.model = nn.Sequential(
                nn.Conv2d(3, 4, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
                nn.ReLU(),
                nn.Linear(32, 4, bias=True)
            )


    def forward(self, x):
        return self.model(x)


"""
Meta-DQN network
"""


class MamlParamsDQN(nn.Module):
    def __init__(self, grid_size):
        super(MamlParamsDQN, self).__init__()

        self.img_reduced_dim = grid_size + 2 - 2 * 4

        self.theta_shapes = [[4, 3, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                             [4, 32], [4]]

        # self.batch_norm1 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm2 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm3 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        # self.batch_norm4 = nn.BatchNorm2d(self.filters, track_running_stats=False)
        #
        # self.max_pool = nn.MaxPool2d(2)
        #
        # self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.theta_shapes))

        self.theta_0 = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
        for i in range(len(self.theta_0)):
            if self.theta_0[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta_0[i])

    def get_theta(self):
        return self.theta_0

    def get_size(self):
        return np.sum([np.prod(x) for x in self.theta_shapes])

    def forward(self, x, theta=None):

        if theta is None:
            theta = self.theta_0

        h = F.relu(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
        h = F.relu(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
        h = F.relu(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
        h = F.relu(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
        h = h.contiguous()
        h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        h = F.relu(F.linear(h, theta[8], bias=theta[9]))
        y = F.linear(h, theta[10], bias=theta[11])

        return y


"""
Meta-PG network
"""


class MamlParamsPg(nn.Module):

    def __init__(self, grid_size, show_goal, lr, model_type=0, activation='tanh', adaptive_lr=False):
        super(MamlParamsPg, self).__init__()

        self.img_reduced_dim = grid_size + 2 - 2 * 4
        if show_goal == 0:
            n_filters = 2
        else:
            n_filters = 3

        if activation == 'tanh':
            self.activation_f = nn.Tanh()
        elif activation == 'relu':
            self.activation_f = nn.ReLU()

        if model_type == 0:
            self.theta_shapes = [[4, n_filters, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [4, 32], [4],
                                 ]
            self.f_theta = self.f_theta0
        elif model_type == 1:
            self.theta_shapes = [[4, n_filters, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [32, 32], [32],
                                 [4, 32], [4],
                                 ]
            self.f_theta = self.f_theta1

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

    def f_theta0(self, x, theta=None):

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

        return pi

    def f_theta1(self, x, theta=None):

        if theta is None:
            theta = self.theta_0

        h = self.activation_f(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
        h = h.contiguous()
        h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        h_a = self.activation_f(F.linear(h, theta[8], bias=theta[9]))
        h_a = self.activation_f(F.linear(h_a, theta[10], bias=theta[11]))
        h_a = F.linear(h_a, theta[12], bias=theta[13])
        pi = F.softmax(h_a, -1)

        return pi

    def forward(self, x, theta=None):

        if theta is None:
            theta = self.theta_0

        pi = self.f_theta(x, theta=theta)

        dist = Categorical(pi)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob, None

    def evaluate(self, state, action, theta=None):

        if theta is None:
            theta = self.theta_0

        pi = self.f_theta(state, theta=theta)

        if torch.is_tensor(action):
            action = action.tolist()
        else:
            action = [action]

        action_logprobs = torch.log(torch.clamp(pi[range(len(action)), action], min=1e-10)) #dist.log_prob(action)

        return action_logprobs, None, None
'''
Maml-PPO
'''

class MamlParamsPPO(nn.Module):

    def __init__(self, grid_size, show_goal, lr, model_type=0, activation='tanh', adaptive_lr=False):
        super(MamlParamsPPO, self).__init__()

        self.img_reduced_dim = grid_size + 2 - 2 * 4
        if show_goal == 0:
            n_filters = 2
        else:
            n_filters = 3

        if activation == 'tanh':
            self.activation_f = nn.Tanh()
        elif activation == 'relu':
            self.activation_f = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_f = nn.LeakyReLU()

        if model_type == 0:
            self.theta_shapes = [[4, n_filters, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [4, 32], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [1, 32], [1],
                                 ]
            self.f_theta = self.f_theta0
        elif model_type == 1:
            self.theta_shapes = [[4, n_filters, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [32, 32], [32],
                                 [4, 32], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [32, 32], [32],
                                 [1, 32], [1],
                                 ]
            self.f_theta = self.f_theta1

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

    def f_theta0(self, x, theta=None):

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

    def f_theta1(self, x, theta=None):

        if theta is None:
            theta = self.theta_0

        h = self.activation_f(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
        h = h.contiguous()
        h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        h_a = self.activation_f(F.linear(h, theta[8], bias=theta[9]))
        h_a = self.activation_f(F.linear(h_a, theta[10], bias=theta[11]))
        h_a = F.linear(h_a, theta[12], bias=theta[13])
        pi = F.softmax(h_a, -1)
        h_c = self.activation_f(F.linear(h, theta[14], bias=theta[15]))
        h_c = self.activation_f(F.linear(h_c, theta[16], bias=theta[17]))
        h_c = F.linear(h_c, theta[18], bias=theta[19])
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


class ActorCriticMAML(nn.Module):
    """
    Ex self.policy = ActorCritic(state_dim, action_dim).to(device)
    """

    def __init__(self, grid_size, show_goal, lr, theta_i, model_type=0, activation='tanh', adaptive_lr=False):
        super(ActorCriticMAML, self).__init__()

        self.img_reduced_dim = grid_size + 2 - 2 * 4
        if show_goal == 0:
            n_filters = 2
        else:
            n_filters = 3

        if activation == 'tanh':
            self.activation_f = nn.Tanh()
        elif activation == 'relu':
            self.activation_f = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_f = nn.LeakyReLU()

        if model_type == 0:
            self.theta_shapes = [[4, n_filters, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [4, 32], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [1, 32], [1],
                                 ]
            self.f_theta = self.f_theta0
        elif model_type == 1:
            self.theta_shapes = [[4, n_filters, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [4, 4, 3, 3], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [32, 32], [32],
                                 [4, 32], [4],
                                 [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                                 [32, 32], [32],
                                 [1, 32], [1],
                                 ]
            self.f_theta = self.f_theta1

        if adaptive_lr:
            self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.theta_shapes))
        else:
            self.lr = [lr] * len(self.theta_shapes)

        self.theta_0 = nn.ParameterList([nn.Parameter(theta) for theta in theta_i])

    def f_theta0(self, x, theta=None):

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

    def f_theta1(self, x, theta=None):

        if theta is None:
            theta = self.theta_0

        h = self.activation_f(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
        h = h.contiguous()
        h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        h_a = self.activation_f(F.linear(h, theta[8], bias=theta[9]))
        h_a = self.activation_f(F.linear(h_a, theta[10], bias=theta[11]))
        h_a = F.linear(h_a, theta[12], bias=theta[13])
        pi = F.softmax(h_a, -1)
        h_c = self.activation_f(F.linear(h, theta[14], bias=theta[15]))
        h_c = self.activation_f(F.linear(h_c, theta[16], bias=theta[17]))
        h_c = F.linear(h_c, theta[18], bias=theta[19])
        v = h_c[:, 0]

        return pi, v

    def get_action(self, state):

        pi, _ = self.f_theta(state)

        dist = Categorical(pi)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob

    def evaluate(self, state, action):

        pi, v = self.f_theta(state)
        dist = Categorical(pi)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # TODO: to squeeze or not to squeeze? that is the question
        return action_logprobs, torch.squeeze(v), dist_entropy

"""
PPO network
"""


class ActorCritic(nn.Module):
    """
    Ex self.policy = ActorCritic(state_dim, action_dim).to(device)
    """

    def __init__(self, grid_size, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        img_reduced_dim = grid_size + 2 - 2 * 4

        self.body = nn.Sequential(
            nn.Conv2d(state_dim, 4, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.actor = nn.Sequential(
            nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
            nn.LeakyReLU(),
            nn.Linear(32, action_dim, bias=True),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
            nn.LeakyReLU(),
            nn.Linear(32, 1, bias=True)
        )

    def get_action(self, state):
        action_probs = self.actor(self.body(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob

    def evaluate(self, state, action):
        state_value = self.critic(self.body(state))

        # to calculate action score(logprobs) and distribution entropy
        action_probs = self.actor(self.body(state))
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy


"""
REINFORCE network
"""


class Actor(nn.Module):
    """
    Ex self.policy = ActorCritic(state_dim, action_dim).to(device)
    """

    def __init__(self, grid_size, state_dim, action_dim):
        super(Actor, self).__init__()

        img_reduced_dim = grid_size + 2 - 2 * 4

        self.actor = nn.Sequential(
            nn.Conv2d(state_dim, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, action_dim, bias=True),
            nn.Softmax(dim=-1)
        )

    def get_action(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        # dist_entropy = dist.entropy()

        return action.item(), action_logprob #, dist_entropy

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = torch.log(torch.max(action_probs[0, action], torch.tensor(1e-5))) #dist.log_prob(action)
        # dist_entropy = dist.entropy()


        return action_logprobs#, dist_entropy


"""
Forward Dynamics Predictor network
"""


class Forward_Predictor(nn.Module):

    def __init__(self, grid_size, show_goal):
        super(Forward_Predictor, self).__init__()

        img_reduced_dim = grid_size + 2 - 2 * 4
        if show_goal == 0:
            n_filters = 2
        else:
            n_filters = 3

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(n_filters, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_reduced_dim * img_reduced_dim * 4, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 2, bias=True),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(3, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 2, bias=True),
        )

    def phi(self, x):
        return self.feature_extractor(x)

    def forward(self, x, a):
        x = torch.cat([x, a], -1)
        return self.forward_model(x)
