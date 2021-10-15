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




# class MamlParamsPPO(nn.Module):
#
#     def __init__(self, params, activation='tanh', adaptive_lr=False):
#         super(MamlParamsPPO, self).__init__()
#         grid_size = params['grid_size']
#         show_goal = params['show_goal']
#         lr = params['inner_lr']
#         self.beta_model = params['beta_model']
#
#         self.inner_loss_type = params['inner_loss_type']
#         self.decoupled_reward = params['decoupled_predictors']
#
#         self.img_reduced_dim = grid_size + 2 - 2 * 4
#         if show_goal == 0:
#             n_filters = 2
#         else:
#             n_filters = 3
#
#         if activation == 'tanh':
#             self.activation_f = nn.Tanh()
#         elif activation == 'relu':
#             self.activation_f = nn.ReLU()
#         elif activation == 'leaky_relu':
#             self.activation_f = nn.LeakyReLU()
#
#         self.decouple_explorer = params['decoupled_explorer']
#         if self.decouple_explorer == 1:
#             self.psi_shapes = [[4, n_filters, 3, 3], [4],
#                                [4, 4, 3, 3], [4],
#                                [4, 4, 3, 3], [4],
#                                [4, 4, 3, 3], [4],
#                                [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
#                                [4, 32], [4],
#                                # [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
#                                # [1, 32], [1],
#                               ]
#             self.psi = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.psi_shapes])
#             for i in range(len(self.psi)):
#                 if self.psi[i].dim() > 1:
#                     torch.nn.init.kaiming_uniform_(self.psi[i])
#
#         if self.inner_loss_type == 1:
#             # beta connected with just z_0
#             if self.beta_model == 0:
#                 self.z_shapes = [[4, n_filters, 3, 3], [4],
#                                  [4, 4, 3, 3], [4],
#                                  [4, 4, 3, 3], [4],
#                                  [4, 4, 3, 3], [4],
#                                  ]
#                 self.theta_shapes = [[32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
#                                     [4, 32], [4],
#                                    ]
#                 self.theta_v_shapes =  [
#                                     [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
#                                     [1, 32], [1],
#                                 ]
#                 self.beta_shapes = [[32, (self.img_reduced_dim * self.img_reduced_dim * 4)+1], [32],
#                                    [1, 32], [1],
#                                   ]
#                 self.z_0 = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.z_shapes])
#                 for i in range(len(self.z_0)):
#                     if self.z_0[i].dim() > 1:
#                         torch.nn.init.kaiming_uniform_(self.z_0[i])
#
#             # Beta is connected with theta and z_0
#             if self.beta_model == 1:
#                 self.theta_shapes = [[4, n_filters, 3, 3], [4],
#                                      [4, 4, 3, 3], [4],
#                                      [4, 4, 3, 3], [4],
#                                      [4, 4, 3, 3], [4],
#                                      [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
#                                      [4, 32], [4],
#                                      ]
#                 self.theta_v_shapes = [
#                                     [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
#                                     [1, 32], [1],
#                                 ]
#                 self.beta_shapes = [[32, (self.img_reduced_dim * self.img_reduced_dim * 4) + 1+ 4], [32],
#                                     [1, 32], [1],
#                                     ]
#             self.theta = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
#             self.theta_v = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_v_shapes])
#             self.beta = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.beta_shapes])
#
#             for i in range(len(self.theta)):
#                 if self.theta[i].dim() > 1:
#                     torch.nn.init.kaiming_uniform_(self.theta[i])
#             for i in range(len(self.theta_v)):
#                 if self.theta_v[i].dim() > 1:
#                     torch.nn.init.kaiming_uniform_(self.theta_v[i])
#             for i in range(len(self.beta)):
#                 if self.beta[i].dim() > 1:
#                     torch.nn.init.kaiming_uniform_(self.beta[i])
#         else:
#             self.theta_shapes = [[4, n_filters, 3, 3], [4],
#                                  [4, 4, 3, 3], [4],
#                                  [4, 4, 3, 3], [4],
#                                  [4, 4, 3, 3], [4],
#                                  [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
#                                  [4, 32], [4],
#                                  [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
#                                  [1, 32], [1],
#                                  ]
#             self.theta_0 = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
#             for i in range(len(self.theta_0)):
#                 if self.theta_0[i].dim() > 1:
#                     torch.nn.init.kaiming_uniform_(self.theta_0[i])
#
#         if adaptive_lr:
#             self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.theta_shapes))
#         else:
#             self.lr = [lr] * len(self.theta_shapes)
#
#     def get_theta(self):
#         if self.inner_loss_type == 0:
#             return self.theta_0
#         elif self.inner_loss_type == 1:
#             if self.beta_model == 0:
#                 return self.z_0
#             else:
#                 return self.theta                  # TODO: is this a good way to pass the models
#
#     def f_theta(self, x, theta=None):
#
#         if self.inner_loss_type == 0:
#             if theta is None:
#                 theta = self.theta_0
#                 if self.decouple_explorer == 1:
#                     # theta = self.psi
#                     print("ERROR - Wrong combination of parameters")
#                     exit()
#
#             h = self.activation_f(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
#             h = h.contiguous()
#             h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
#             h_a = self.activation_f(F.linear(h, theta[8], bias=theta[9]))
#             h_a = F.linear(h_a, theta[10], bias=theta[11])
#             pi = F.softmax(h_a, -1)
#             h_c = self.activation_f(F.linear(h, theta[12], bias=theta[13]))
#             h_c = F.linear(h_c, theta[14], bias=theta[15])
#             v = h_c[:, 0]
#
#         elif self.inner_loss_type == 1:
#
#             decoupled_exploration = False
#             if theta is None:
#                 if self.beta_model == 0:
#                     theta = self.z_0
#                 else:
#                     theta = self.theta
#                 theta_v = self.theta_v
#                 if self.decouple_explorer == 1:
#                     theta = self.psi
#                     decoupled_exploration = True
#
#
#             h = self.activation_f(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
#             h = h.contiguous()
#             h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
#
#             if decoupled_exploration:
#                 h_a = self.activation_f(F.linear(h, theta[8], bias=theta[9]))
#                 h_a = F.linear(h_a, theta[10], bias=theta[11])
#                 pi = F.softmax(h_a, -1)
#                 # h_c = self.activation_f(F.linear(h, theta[12], bias=theta[13]))
#                 # h_c = F.linear(h_c, theta[14], bias=theta[15])
#                 v = 0
#                 # v = pi[:, 0]*0#h_c[:, 0]
#             else:
#                 if self.beta_model == 0:
#                     h_a = self.activation_f(F.linear(h, self.theta[0], bias=self.theta[1]))
#                     h_a = F.linear(h_a, self.theta[2], bias=self.theta[3])
#                     pi = F.softmax(h_a, -1)
#                 else:
#                     h_a = self.activation_f(F.linear(h, theta[8], bias=theta[9]))
#                     h_a = F.linear(h_a, theta[10], bias=theta[11])
#                     pi = F.softmax(h_a, -1)
#
#                 h_c = self.activation_f(F.linear(h, self.theta_v[0], bias=self.theta_v[1]))
#                 h_c = F.linear(h_c, self.theta_v[2], bias=self.theta_v[3])
#                 v = h_c[:, 0]
#
#         return pi, v
#
#     def get_predicted_reward(self, x, a):
#         if self.beta_model == 0:
#             h = self.activation_f(F.conv2d(x, self.z_0[0], bias=self.z_0[1], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, self.z_0[2], bias=self.z_0[3], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, self.z_0[4], bias=self.z_0[5], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, self.z_0[6], bias=self.z_0[7], stride=1, padding=0))
#             h = h.contiguous()
#             h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
#             h = torch.cat([h, a], -1)
#             h_r = self.activation_f(F.linear(h, self.beta[0], bias=self.beta[1]))
#             h_r = F.linear(h_r, self.beta[2], bias=self.beta[3])
#
#         else:
#             # pass through z_0
#             h = self.activation_f(F.conv2d(x, self.theta[0], bias=self.theta[1], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, self.theta[2], bias=self.theta[3], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, self.theta[4], bias=self.theta[5], stride=1, padding=0))
#             h = self.activation_f(F.conv2d(h, self.theta[6], bias=self.theta[7], stride=1, padding=0))
#             h = h.contiguous()
#             h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
#
#             # pass through theta
#             h_a = self.activation_f(F.linear(h, self.theta[8], bias=self.theta[9]))
#             h_a = F.linear(h_a, self.theta[10], bias=self.theta[11])
#
#             h = torch.cat([h, a, h_a], -1)
#
#             h_r = self.activation_f(F.linear(h, self.beta[0], bias=self.beta[1]))
#             h_r = F.linear(h_r, self.beta[2], bias=self.beta[3])
#
#         return h_r[:, 0]
#
#     def forward(self, x, theta=None):
#
#         pi, v = self.f_theta(x, theta=theta)
#
#         dist = Categorical(pi)
#         action = dist.sample()
#         action_logprob = dist.log_prob(action)
#
#         return action.item(), action_logprob, v
#
#     def evaluate(self, x, action, theta=None):
#
#         pi, v = self.f_theta(x, theta=theta)
#
#         if torch.is_tensor(action):
#             action = action.tolist()
#         else:
#             action = [action]
#
#         action_logprobs = torch.log(torch.clamp(pi[range(len(action)), action], min=1e-10))
#
#         dist = Categorical(pi)
#         dist_entropy = dist.entropy()
#
#         return action_logprobs, v, dist_entropy


class Curiosity2MamlParamsPPO(nn.Module):

    def __init__(self, params, activation='tanh', adaptive_lr=False):
        super(Curiosity2MamlParamsPPO, self).__init__()
        grid_size = params['grid_size']
        show_goal = params['show_goal']
        lr = params['inner_lr']

        self.decoupled_reward = params['decoupled_predictors']

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

        # self.decouple_explorer = params['decoupled_explorer']
        # if self.decouple_explorer == 1:
        #     self.psi_shapes = [[4, n_filters, 3, 3], [4],
        #                        [4, 4, 3, 3], [4],
        #                        [4, 4, 3, 3], [4],
        #                        [4, 4, 3, 3], [4],
        #                        [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
        #                        [4, 32], [4],
        #                       ]
        #     self.psi = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.psi_shapes])
        #     for i in range(len(self.psi)):
        #         if self.psi[i].dim() > 1:
        #             torch.nn.init.kaiming_uniform_(self.psi[i])
        #
        # self.theta_shapes = [[4, n_filters, 3, 3], [4],
        #                      [4, 4, 3, 3], [4],
        #                      [4, 4, 3, 3], [4],
        #                      [4, 4, 3, 3], [4],
        #                      [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
        #                      [4, 32], [4]]
        # self.theta_v_shapes = [[32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
        #                        [1, 32], [1]]
        # self.beta_shapes = [[32, (self.img_reduced_dim * self.img_reduced_dim * 4) + 1 + 4], [32],
        #                     [1, 32], [1]]
        # self.theta = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
        # self.theta_v = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_v_shapes])
        # self.beta = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.beta_shapes])
        self.decouple_explorer = params['decoupled_explorer']
        if self.decouple_explorer == 1:
            self.psi_shapes = [[32, 2*params['path_length']*params['horizon_multiplier_adaptation']], [32],
                               [32, 32], [32],
                               [32, 32], [32],
                               [4, 32], [4],
                               ]
            self.psi = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.psi_shapes])
            for i in range(len(self.psi)):
                if self.psi[i].dim() > 1:
                    torch.nn.init.kaiming_uniform_(self.psi[i])

        self.theta_shapes = [[32, 2*params['path_length']*params['horizon_multiplier_adaptation']], [32],
                             [32, 32], [32],
                             [32, 32], [32],
                             [4, 32], [4],
                             ]
        self.theta_v_shapes = [[32, 32], [32],
                               [1, 32], [1]]
        self.beta_shapes = [[32, 32 + 1 + 4], [32],
                            [1, 32], [1]]
        self.theta = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
        self.theta_v = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_v_shapes])
        self.beta = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.beta_shapes])

        for i in range(len(self.theta)):
            if self.theta[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta[i])
        for i in range(len(self.theta_v)):
            if self.theta_v[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta_v[i])
        for i in range(len(self.beta)):
            if self.beta[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.beta[i])

        if self.decoupled_reward == 0:
            self.beta_tmp = nn.ParameterList([nn.Parameter(x.detach().clone()) for x in self.beta])

        if adaptive_lr:
            self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.theta_shapes))
        else:
            self.lr = [lr] * len(self.theta_shapes)

        self.params_to_adapt = [param for param in self.theta.parameters()]

    def get_explorer_params(self):
        if self.decouple_explorer:
            return self.psi
        return self.theta

    def get_exploiter_starting_params(self):
        return self.theta

    def f_theta(self, x, theta=None):

        decoupled_exploration = False
        if theta is None:
            theta = self.theta
            if self.decouple_explorer == 1:
                theta = self.psi
                decoupled_exploration = True

        # h = self.activation_f(F.conv2d(x, theta[0], bias=theta[1], stride=1, padding=0))
        # h = self.activation_f(F.conv2d(h, theta[2], bias=theta[3], stride=1, padding=0))
        # h = self.activation_f(F.conv2d(h, theta[4], bias=theta[5], stride=1, padding=0))
        # h = self.activation_f(F.conv2d(h, theta[6], bias=theta[7], stride=1, padding=0))
        # h = h.contiguous()
        # h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        # h_a = self.activation_f(F.linear(h, theta[8], bias=theta[9]))
        # h_a = F.linear(h_a, theta[10], bias=theta[11])
        # pi = F.softmax(h_a, -1)
        h = self.activation_f(F.linear(x, theta[0], bias=theta[1]))
        h = self.activation_f(F.linear(h, theta[2], bias=theta[3]))
        h = self.activation_f(F.linear(h, theta[4], bias=theta[5]))
        h_a = self.activation_f(F.linear(h, theta[6], bias=theta[7]))

        pi = F.softmax(h_a, -1)

        if decoupled_exploration:
            v = 0
        else:
            h_c = self.activation_f(F.linear(h, self.theta_v[0], bias=self.theta_v[1]))
            h_c = F.linear(h_c, self.theta_v[2], bias=self.theta_v[3])
            v = h_c[:, 0]

        return pi, v

    def get_predicted_reward(self, x, a, use_beta=False):

        beta = self.beta
        if not use_beta and self.decoupled_reward == 0:
            beta = self.beta_tmp

        # # pass through z_0
        # h = self.activation_f(F.conv2d(x, self.theta[0], bias=self.theta[1], stride=1, padding=0))
        # h = self.activation_f(F.conv2d(h, self.theta[2], bias=self.theta[3], stride=1, padding=0))
        # h = self.activation_f(F.conv2d(h, self.theta[4], bias=self.theta[5], stride=1, padding=0))
        # h = self.activation_f(F.conv2d(h, self.theta[6], bias=self.theta[7], stride=1, padding=0))
        # h = h.contiguous()
        # h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        #
        # # pass through theta
        # h_a = self.activation_f(F.linear(h, self.theta[8], bias=self.theta[9]))
        # h_a = F.linear(h_a, self.theta[10], bias=self.theta[11])
        #
        # h = torch.cat([h, a, h_a], -1)
        #
        # h_r = self.activation_f(F.linear(h, beta[0], bias=beta[1]))
        # h_r = F.linear(h_r, beta[2], bias=beta[3])

        # pass through z_0
        h = self.activation_f(F.linear(x, self.theta[0], bias=self.theta[1]))
        h = self.activation_f(F.linear(h, self.theta[2], bias=self.theta[3]))
        h = self.activation_f(F.linear(h, self.theta[4], bias=self.theta[5]))
        h_a = self.activation_f(F.linear(h, self.theta[6], bias=self.theta[7]))

        h = torch.cat([h, a, h_a], -1)

        h_r = self.activation_f(F.linear(h, beta[0], bias=beta[1]))
        h_r = F.linear(h_r, beta[2], bias=beta[3])

        return h_r[:, 0]

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

    def get_adapt_loss(self, logprobs, old_logprobs, rt, R, st, at, V, c1):

        rt_hat = self.get_predicted_reward(st, at)
        R_err = ((rt - rt_hat) ** 2)

        loss_pi = (- torch.exp(logprobs - old_logprobs.detach()) * R_err).mean()
        loss = loss_pi

        return loss, loss_pi.detach().cpu().item(), 0, R_err.mean().detach().cpu().item()


class Curiosity1MamlParamsPPO(nn.Module):

    def __init__(self, params, activation='tanh', adaptive_lr=False):
        super(Curiosity1MamlParamsPPO, self).__init__()
        grid_size = params['grid_size']
        show_goal = params['show_goal']
        lr = params['inner_lr']

        self.decoupled_reward = params['decoupled_predictors']

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

        self.decouple_explorer = params['decoupled_explorer']
        if self.decouple_explorer == 1:
            self.psi_shapes = [[4, n_filters, 3, 3], [4],
                               [4, 4, 3, 3], [4],
                               [4, 4, 3, 3], [4],
                               [4, 4, 3, 3], [4],
                               [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                               [4, 32], [4]]
            self.psi = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.psi_shapes])
            for i in range(len(self.psi)):
                if self.psi[i].dim() > 1:
                    torch.nn.init.kaiming_uniform_(self.psi[i])

        self.z_shapes = [[4, n_filters, 3, 3], [4],
                         [4, 4, 3, 3], [4],
                         [4, 4, 3, 3], [4],
                         [4, 4, 3, 3], [4]]
        self.theta_shapes = [[32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                             [4, 32], [4]]
        self.theta_v_shapes = [[32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                               [1, 32], [1]]
        self.beta_shapes = [[32, (self.img_reduced_dim * self.img_reduced_dim * 4)+1], [32],
                            [1, 32], [1]]
        self.z_0 = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.z_shapes])
        self.theta = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
        self.theta_v = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_v_shapes])
        self.beta = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.beta_shapes])

        for i in range(len(self.z_0)):
            if self.z_0[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.z_0[i])
        for i in range(len(self.theta)):
            if self.theta[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta[i])
        for i in range(len(self.theta_v)):
            if self.theta_v[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta_v[i])
        for i in range(len(self.beta)):
            if self.beta[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.beta[i])

        if self.decoupled_reward == 0:
            self.beta_tmp = nn.ParameterList([nn.Parameter(x.detach().clone()) for x in self.beta])

        if adaptive_lr:
            self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.z_shapes))
        else:
            self.lr = [lr] * len(self.z_shapes)

        self.params_to_adapt = [param for param in self.z_0.parameters()] + [param for param in self.theta.parameters()]

    def get_explorer_params(self):
        if self.decouple_explorer:
            return self.psi
        return self.z_0

    def get_exploiter_starting_params(self):
        return self.z_0

    def f_theta(self, x, theta=None):

        decoupled_exploration = False
        if theta is None:
            z = self.z_0
            theta = self.theta
            if self.decouple_explorer == 1:
                z = self.psi[:-4]
                theta = self.psi[-4:]
                decoupled_exploration = True
        else:
            z = theta
            theta = self.theta

        h = self.activation_f(F.conv2d(x, z[0], bias=z[1], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, z[2], bias=z[3], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, z[4], bias=z[5], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, z[6], bias=z[7], stride=1, padding=0))
        h = h.contiguous()
        h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        h_a = self.activation_f(F.linear(h, theta[0], bias=theta[1]))
        h_a = F.linear(h_a, theta[2], bias=theta[3])
        pi = F.softmax(h_a, -1)

        if decoupled_exploration:
            v = 0
        else:
            h_c = self.activation_f(F.linear(h, self.theta_v[0], bias=self.theta_v[1]))
            h_c = F.linear(h_c, self.theta_v[2], bias=self.theta_v[3])
            v = h_c[:, 0]

        return pi, v

    def get_predicted_reward(self, x, a, use_beta=False):

        beta = self.beta
        if not use_beta and self.decoupled_reward == 0:
            beta = self.beta_tmp

        h = self.activation_f(F.conv2d(x, self.z_0[0], bias=self.z_0[1], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, self.z_0[2], bias=self.z_0[3], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, self.z_0[4], bias=self.z_0[5], stride=1, padding=0))
        h = self.activation_f(F.conv2d(h, self.z_0[6], bias=self.z_0[7], stride=1, padding=0))
        h = h.contiguous()
        h = h.view(-1, (self.img_reduced_dim * self.img_reduced_dim * 4))
        h = torch.cat([h, a], -1)
        h_r = self.activation_f(F.linear(h, beta[0], bias=beta[1]))
        h_r = F.linear(h_r, beta[2], bias=beta[3])

        return h_r[:, 0]

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

    def get_adapt_loss(self, logprobs, old_logprobs, rt, R, st, at, V, c1):

        rt_hat = self.get_predicted_reward(st, at)
        R_err = ((rt - rt_hat) ** 2)

        loss_pi = (- torch.exp(logprobs - old_logprobs.detach()) * R_err).mean()
        loss = loss_pi

        return loss, loss_pi.detach().cpu().item(), 0, R_err.mean().detach().cpu().item()


class NormalMamlParamsPPO(nn.Module):

    def __init__(self, params, activation='tanh', adaptive_lr=False):
        super(NormalMamlParamsPPO, self).__init__()
        grid_size = params['grid_size']
        show_goal = params['show_goal']
        lr = params['inner_lr']

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

        self.theta_shapes = [[4, n_filters, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [4, 4, 3, 3], [4],
                             [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                             [4, 32], [4],
                             [32, (self.img_reduced_dim * self.img_reduced_dim * 4)], [32],
                             [1, 32], [1],
                             ]
        self.theta_0 = nn.ParameterList([nn.Parameter(torch.zeros(t_size)) for t_size in self.theta_shapes])
        for i in range(len(self.theta_0)):
            if self.theta_0[i].dim() > 1:
                torch.nn.init.kaiming_uniform_(self.theta_0[i])

        if adaptive_lr:
            self.lr = nn.ParameterList([nn.Parameter(torch.tensor(lr))] * len(self.theta_shapes))
        else:
            self.lr = [lr] * len(self.theta_shapes)

    def get_explorer_params(self):
        return self.theta_0

    def get_exploiter_starting_params(self):
        return self.theta_0

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

    def get_adapt_loss(self, logprobs, old_logprobs, rt, R, st, at, V, c1):
        A = R-V

        loss_pi = (- torch.exp(logprobs - old_logprobs.detach()) * A.detach()).mean()  # (- rtgs * logprobs).mean() # TODO: importance sampling / something to update psi
        loss_v = (c1 * A ** 2).mean()
        loss = loss_pi + loss_v

        return loss, loss_pi.detach().cpu().item(), loss_v.detach().cpu().item(), 0.






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


class Reward_Predictor(nn.Module):

    def __init__(self, grid_size, show_goal):
        super(Reward_Predictor, self).__init__()

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
            nn.Linear(32, 32, bias=True),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(32+1, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 1, bias=True),
        )

    def phi(self, x):
        return self.feature_extractor(x)

    def forward(self, s, a):
        z = self.feature_extractor(s)
        x = torch.cat([z, a], -1)
        return torch.squeeze(self.forward_model(x))
