import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim import Adam, RMSprop, SGD
from RL_Project.networks import MamlParamsPPO
from RL_Project.utils import BatchData, calc_rtg
# from networks import MamlParamsPPO
# from utils import BatchData, calc_rtg

"""

PPO meta-agent

"""

class PPO(nn.Module):
    def __init__(self, params, logdir, device, adaptive_lr=False):
        super(PPO, self).__init__()

        self.batchdata = [[BatchData() for _ in range(params['batch_tasks'])], [BatchData() for _ in range(params['batch_tasks'])]]        # list[0] for adaptation batch and list[1] for evaluation batch
        self.writer = SummaryWriter(log_dir=logdir)
        self.log_idx = 0
        self.log_grads_idx = 0
        self.device = device

        self.inner_lr = 0.1

        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_adapt = params['eps_adapt'] #1.0 #0.5
        self.epsilon_decay = 0.8 #0.85

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr = 0.0001
        self.eps_clip = 0.1
        self.gamma = 0.9
        self.c1 = params['c1']
        self.c2 = params['c2']

        self.norm_A = params['norm_A']

        self.policy = MamlParamsPPO(10, self.inner_lr, adaptive_lr=adaptive_lr)

        self.grads_vals = np.zeros(len(self.policy.get_theta()))

        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)

    def get_action(self, state, theta=None, test=False):
        # Sample actions with epsilon greedy
        if np.random.random() > self.epsilon or test:
            a, log_prob, v = self.policy(self.to_tensor(state), theta=theta)
            return a, log_prob, v
        else:
            a = np.random.randint(0, 4)
            log_prob, v, _ = self.policy.evaluate(self.to_tensor(state), a, theta=theta)
            return a, log_prob, v

    def adapt(self, idx, train=False, print_grads=False):

        self.policy.train()

        theta_i = self.policy.get_theta()

        rtgs = self.to_tensor(calc_rtg(self.batchdata[0][idx].rewards, self.batchdata[0][idx].is_terminal, self.gamma))  # reward-to-go
        # Normalize rewards
        logprobs = torch.cat([x.view(1) for x in self.batchdata[0][idx].logprobs])
        v = torch.cat([x.view(1) for x in self.batchdata[0][idx].v])

        loss_pi = (- rtgs * logprobs).mean()
        loss_v = (self.c1*(v - rtgs)**2).mean()
        loss = loss_pi + loss_v

        if train:
            theta_grad_s = torch.autograd.grad(outputs=loss, inputs=theta_i, create_graph=True)
            theta_i = list(map(lambda p: p[1] - p[2] * p[0], zip(theta_grad_s, theta_i, self.policy.lr)))

            if print_grads:
                for i, grad in enumerate(theta_grad_s):
                    self.grads_vals[i] += torch.mean(torch.abs(grad))

        else:
            theta_grad_s = torch.autograd.grad(outputs=loss, inputs=theta_i)
            theta_i = list(map(lambda p: p[1] - p[2] * p[0].detach(), zip(theta_grad_s, theta_i, self.policy.lr)))

        return theta_i, loss.detach().cpu().item(), loss_pi.detach().cpu().item(), loss_v.detach().cpu().item()

    def update_adaptation_batches(self):

        for batchdata in (self.batchdata[0]):
            states = torch.cat([self.to_tensor(x) for x in batchdata.states], 0).detach()
            actions = self.to_tensor(batchdata.actions).long().detach()
            logprobs, state_vals, _ = self.policy.evaluate(states, actions)

            batchdata.logprobs = [x for x in logprobs]
            batchdata.v = [x for x in state_vals]

    def get_loss(self, theta_i, idx):
        # get form correct batch old policy data
        rtgs = self.to_tensor(calc_rtg(self.batchdata[1][idx].rewards, self.batchdata[1][idx].is_terminal, self.gamma))  # reward-to-go

        old_states = torch.cat([self.to_tensor(x) for x in self.batchdata[1][idx].states], 0).detach()
        old_actions = self.to_tensor(self.batchdata[1][idx].actions).long().detach()
        old_logprobs = torch.cat([x.view(1) for x in self.batchdata[1][idx].logprobs]).detach()

        #get form correct batch new policy data
        logprobs, state_vals, H = self.policy.evaluate(old_states, old_actions, theta=theta_i)

        # Compute loss
        # Importance ratio
        ratios = torch.exp(logprobs - old_logprobs.detach())  # new probs over old probs

        # Calc advantages
        A = rtgs - state_vals
        if self.norm_A == 1:
            A = ((A - torch.mean(A)) / torch.std(A)).detach()

        # Actor loss using CLIP loss
        surr1 = ratios * A
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * A
        actor_loss = torch.mean( - torch.min(surr1, surr2) )  # minus to maximize

        # Critic loss fitting to reward-to-go with entropy bonus
        critic_loss = self.c1 * self.MSE_loss(rtgs, state_vals)

        loss = actor_loss + critic_loss - self.c2 * H.mean()

        return loss

    def push_batchdata(self, st, a, logprob, v, r, done, mode, idx):
        # adds a row of trajectory data to self.batchdata
        self.batchdata[mode][idx].states.append(st)
        self.batchdata[mode][idx].actions.append(a)
        self.batchdata[mode][idx].logprobs.append(logprob)
        self.batchdata[mode][idx].v.append(v)
        self.batchdata[mode][idx].rewards.append(r)
        self.batchdata[mode][idx].is_terminal.append(done)

    def clear_batchdata(self):
        for i in range(2):
            for batchdata in self.batchdata[i]:
                batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(self.device)
        else:
            return torch.tensor(array, dtype=torch.float).to(self.device)
