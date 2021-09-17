import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random
from networks import MamlParamsPg, MamlParamsPPO, Forward_Predictor, ActorCritic, ActorCriticMAML
from utils import BatchData, calc_rtg
import numpy as np
from torch.optim import Adam, RMSprop, SGD

"""

PPO meta-agent

"""

class Meta_PPO(nn.Module):
    def __init__(self, params, device, adaptive_lr=False, load_pretrained=False):
        super(Meta_PPO, self).__init__()

        self.action_dim = 4
        self.show_goal = params['show_goal']
        self.batchdata = BatchData() # for _ in range(params['batch_tasks'])]        # list[0] for adaptation batch and list[1] for evaluation batch

        self.log_idx = 0
        self.log_grads_idx = 0
        self.device = device

        self.inner_lr = 0.1

        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_adapt = 1.0 #0.9
        self.epsilon_adapt_decay = 1.0 #params['eps_adapt_decay']
        self.epsilon_decay = 0.98 #0.85

        self.K = params['episode_per_update']

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr = params['lr']
        self.eps_clip = params['eps_clip']
        self.gamma = 0.9
        self.c1 = params['c1']
        self.c2 = params['c2']

        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.nu = params['kl_nu']

        self.ADAPTATION = 0
        self.EVALUATION = 1

        self.norm_A = params['norm_A']

        self.policy = MamlParamsPPO(params['grid_size'], self.show_goal, self.inner_lr, 0, 0, activation=params['activation'], adaptive_lr=adaptive_lr)

        self.grads_vals = np.zeros(len(self.policy.get_theta()))

        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, eps=params['adam_epsilon'])

        if params['filter_type'] == 4:
            self.forward_model = Forward_Predictor(params['grid_size'], self.show_goal)
            self.optimizer_forward = Adam(self.forward_model.parameters(), lr=0.0001)

    def get_epsilon(self, mode=0):
        if mode == self.ADAPTATION:
            return self.epsilon_adapt
        else:
            return self.epsilon0

    def update_epsilon(self, mode):
        if mode == self.EVALUATION:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_adapt_decay

    def get_action(self, state, theta=None, test=False):
        # Sample actions with epsilon greedy
        if np.random.random() > self.epsilon or test:
            a, log_prob, v = self.policy(self.to_tensor(state), theta=theta)
            return a, log_prob, v
        else:
            a = np.random.randint(0, self.action_dim)
            log_prob, v, _ = self.policy.evaluate(self.to_tensor(state), a, theta=theta)
            return a, log_prob, v

    def adapt(self, train=False, print_grads=False):

        self.policy.train()

        theta_i = self.policy.get_theta()

        states = torch.cat([self.to_tensor(x) for x in self.batchdata.states], 0).detach()
        actions = self.to_tensor(self.batchdata.actions).view(-1, 1).detach()
        old_logprobs = torch.cat([x.view(1) for x in self.batchdata.old_logprobs])
        rtgs = self.to_tensor(calc_rtg(self.batchdata.rewards, self.batchdata.is_terminal, self.gamma))  # reward-to-go
        # Normalize rewards
        logprobs, v, _ = self.policy.evaluate(states, actions[:,0], theta=theta_i)

        last_trj = int(len(self.batchdata.states) * 9 / 10)
        visited_states = np.concatenate(self.batchdata.states[last_trj:], 0)
        img = np.sum(visited_states, 0)[1]

        A = rtgs - v

        loss_pi = (- torch.exp(logprobs - old_logprobs.detach()) * A.detach()).mean()  # (- rtgs * logprobs).mean() #
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

        return theta_i, loss.detach().cpu().item(), loss_pi.detach().cpu().item(), loss_v.detach().cpu().item(), img

    # def update_adaptation_batches(self):
    #
    #     for batchdata in (self.batchdata[0]):
    #         states = torch.cat([self.to_tensor(x) for x in batchdata.states], 0).detach()
    #         actions = self.to_tensor(batchdata.actions).long().detach()
    #         logprobs, state_vals, _ = self.policy.evaluate(states, actions)
    #
    #         batchdata.logprobs = [x for x in logprobs]
    #         batchdata.v = [x for x in state_vals]

    # def get_loss(self, theta_i):
    #     # get form correct batch old policy data
    #     rtgs = self.to_tensor(calc_rtg(self.batchdata.rewards, self.batchdata.is_terminal, self.gamma))  # reward-to-go
    #
    #     old_states = torch.cat([self.to_tensor(x) for x in self.batchdata[1][idx].states], 0).detach()
    #     old_actions = self.to_tensor(self.batchdata[1][idx].actions).long().detach()
    #     old_logprobs = torch.cat([x.view(1) for x in self.batchdata[1][idx].logprobs]).detach()
    #
    #     #get form correct batch new policy data
    #     logprobs, state_vals, H = self.policy.evaluate(old_states, old_actions, theta=theta_i)
    #
    #     # Compute loss
    #     # Importance ratio
    #     ratios = torch.exp(logprobs - old_logprobs.detach())  # new probs over old probs
    #
    #     # Calc advantages
    #     A = rtgs - state_vals
    #     if self.norm_A == 1:
    #         A = ((A - torch.mean(A)) / torch.std(A)).detach()
    #
    #     # Actor loss using CLIP loss
    #     surr1 = ratios * A
    #     surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * A
    #     actor_loss = torch.mean( - torch.min(surr1, surr2) )  # minus to maximize
    #
    #     # Critic loss fitting to reward-to-go with entropy bonus
    #     critic_loss = self.c1 * self.MSE_loss(rtgs, state_vals)
    #
    #     loss = actor_loss + critic_loss - self.c2 * H.mean()
    #
    #     return loss




    def save_model(self, filepath='./ppo_model.pth'):  # TODO filename param
        torch.save(self.policy.state_dict(), filepath)

    def load_model(self, filepath='./ppo_model.pth'):
        self.policy.load_state_dict(torch.load(filepath))

    # def write_reward(self, r, r2):
    #     """
    #     Function that write on tensorboard the rewards it gets
    #
    #     :param r: cumulative reward of the episode
    #     :type r: float
    #     :param r2: final reword of the episode
    #     :type r2: float
    #     """
    #     self.writer.add_scalar('cumulative_reward', r, self.log_idx)
    #     self.writer.add_scalar('final_reward', r2, self.log_idx)
    #     self.log_idx += 1

    def push_batchdata(self, st, a, logprob, old_logprob, v, r, done, st1):
        # adds a row of trajectory data to self.batchdata
        self.batchdata.states.append(st)
        self.batchdata.actions.append(a)
        self.batchdata.logprobs.append(logprob.detach())
        self.batchdata.old_logprobs.append(old_logprob.detach())
        self.batchdata.v.append(v if isinstance(v, int) else v.detach())
        self.batchdata.rewards.append(r)
        self.batchdata.is_terminal.append(done)
        self.batchdata.next_states.append(st1)

    def clear_batchdata(self):
        self.batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(self.device)
        else:
            return torch.tensor(array, dtype=torch.float).to(self.device)

class PPO(nn.Module):

    def __init__(self, params, logdir, device, theta_i):
        super(PPO, self).__init__()
        # extract environment info from maze....
        # self.mazesim = mazesim
        self.state_dim = 3  # I guess for 1 grid image?
        if params['show_goal'] == 0:
            self.state_dim = 2  # I guess for 1 grid image?

        self.action_dim = 4  # {0: Down, 1: Up, 2: Right, 3: Left}
        self.batchdata = BatchData()
        # self.writer = SummaryWriter(log_dir=logdir)
        self.log_idx = 0
        self.device = device

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr = 0.001
        self.eps_clip = 0.1
        self.gamma = 0.9
        self.c1 = 0.5  # VF loss coefficient
        self.c2 = 0.1  # Entropy bonus coefficient
        self.K_epochs = 10  # num epochs to train on batch data
        self.epsilon = 0.9

        self.policy = ActorCriticMAML(params['grid_size'], params['show_goal'], 0.1, theta_i, model_type=params['model_type'], activation=params['activation']).to(device)


        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.policy_optim = RMSprop(self.policy.parameters(), self.lr)
        # self.policy_optim = Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': self.lr},
        #     {'params': self.policy.critic.parameters(), 'lr': self.lr}
        # ])

        self.old_policy = ActorCriticMAML(params['grid_size'], params['show_goal'], 0.1, theta_i, model_type=params['model_type'], activation=params['activation']).to(device)
        # self.old_policy.load_state_dict(self.policy.state_dict())

    def get_action(self, state, test=False):
        # Sample actions from 'old policy'
        # if np.random.random() > self.epsilon:
        #     return self.old_policy.get_action(self.to_tensor(state))
        # else:
        #     a = np.random.randint(0, 4)
        #     return a, self.old_policy.evaluate(self.to_tensor(state), torch.tensor(a).to(self.device))[0]
        return self.old_policy.get_action(self.to_tensor(state))

    #     if(random.random() > self.epsilon or test):
    #         a = self.policy.act

    def update(self):
        """
            Updates the actor-critic networks for current batch data
        """
        rtgs = self.to_tensor(calc_rtg(self.batchdata.rewards, self.batchdata.is_terminal, self.gamma))  # reward-to-go
        # Normalize rewards
        # rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-5)   # todo: ?

        old_states = torch.cat([self.to_tensor(x) for x in self.batchdata.states], 0).detach()
        old_actions = self.to_tensor(self.batchdata.actions).detach()
        old_logprobs = self.to_tensor(self.batchdata.logprobs).detach() # todo: check actions


        # Train policy for K epochs on collected trajectories, sample and update
        # Evaluate old actions and values using current policy
        for _ in range(self.K_epochs):
            logprobs, state_vals, dist_entropy = self.policy.evaluate(old_states, old_actions)  # todo: these logprobs could not work

            # Importance ratio
            ratios = torch.exp(logprobs - old_logprobs.detach())  # new probs over old probs

            # Calc advantages
            A = rtgs - state_vals.detach()  # old rewards and old states evaluated by curr policy
            A = ((A - torch.mean(A)) / torch.std(A)).detach()

            # Normalize advantages
            # advantages = (A-A.mean()) / (A.std() + 1e-5)

            # Actor loss using CLIP loss
            surr1 = ratios * A
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * A
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # minus to maximize

            # Critic loss fitting to reward-to-go with entropy bonus
            critic_loss = self.c1 * self.MSE_loss(rtgs, state_vals) - self.c2 * torch.mean(dist_entropy)

            loss = actor_loss + critic_loss

            self.policy_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optim.step()

        # Replace old policy with new policy
        self.old_policy.load_state_dict(self.policy.state_dict())

    def save_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):  # TODO filename param
        torch.save(self.policy.actor.state_dict(), actor_filepath)
        torch.save(self.policy.critic.state_dict(), critic_filepath)

    def load_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):
        self.policy.actor.load_state_dict(torch.load(actor_filepath))
        self.policy.critic.load_state_dict(torch.load(critic_filepath))

    # def write_reward(self, r, offset):
    #     """
    #     Function that write on tensorboard the rewards it gets
    #
    #     :param r: cumulative reward of the episode
    #     :type r: float
    #     :param r2: final reword of the episode
    #     :type r2: float
    #     """
    #     self.writer.add_scalar('final_reward', r, self.log_idx+offset)
    #     self.log_idx += 1

    def push_batchdata(self, st, a, logprob, r, done):
        # adds a row of trajectory data to self.batchdata
        self.batchdata.states.append(st)
        self.batchdata.actions.append(a)
        self.batchdata.logprobs.append(logprob.detach())
        self.batchdata.rewards.append(r)
        self.batchdata.is_terminal.append(done)

    def clear_batchdata(self):
        self.batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(self.device)
        else:
            return torch.tensor(array, dtype=torch.float).to(self.device)