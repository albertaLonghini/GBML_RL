import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim import Adam, RMSprop
from RL_Project.networks import ActorCritic, Actor
from RL_Project.utils import BatchData, calc_rtg
# from networks import ActorCritic, Actor
# from utils import BatchData, calc_rtg


"""

PPO agent

"""
class PPO(nn.Module):

    def __init__(self, params, logdir, device):
        super(PPO, self).__init__()

        self.batchdata = BatchData()
        self.writer = SummaryWriter(log_dir=logdir)
        self.log_idx = 0
        self.device = device

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        lr = 0.001
        self.eps_clip = 0.1
        self.gamma = 0.9
        self.c1 = 0.5  # VF loss coefficient
        self.c2 = 0.1  # Entropy bonus coefficient
        self.K_epochs = 10  # num epochs to train on batch data

        self.episode_per_update = params['episode_per_update']

        # epsilon is not used in PPO so these two variables are not really useful
        self.epsilon = 0.9
        self.epsilon0 = 0.9

        self.policy = ActorCritic(10).to(device)
        # if load_pretrained:  # if load actor-critic network params from file
        #     self.load_model()
        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.policy_optim = RMSprop(self.policy.parameters(), lr=lr)

        self.old_policy = ActorCritic(10).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

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

    def push_data(self, s, a, logprob, r, done, s1):
        self.push_batchdata(s, a, logprob, r, done)

    def update(self, epoch):

        if epoch % self.episode_per_update == self.episode_per_update-1:
            self.update_Pi()
            self.clear_batchdata()  # reset the sampled policy trajectories

    def update_Pi(self):
        rtgs = self.to_tensor(calc_rtg(self.batchdata.rewards, self.batchdata.is_terminal, self.gamma))  # reward-to-go
        # Normalize rewards
        # rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-5)

        old_states = torch.cat([self.to_tensor(x) for x in self.batchdata.states], 0).detach()
        old_actions = self.to_tensor(self.batchdata.actions).detach()
        old_logprobs = self.to_tensor(self.batchdata.logprobs).detach()

        for _ in range(self.K_epochs):
            logprobs, state_vals, dist_entropy = self.policy.evaluate(old_states, old_actions)

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

    # def save_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):  # TODO filename param
    #     torch.save(self.policy.actor.state_dict(), actor_filepath)
    #     torch.save(self.policy.critic.state_dict(), critic_filepath)
    #
    # def load_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):
    #     self.policy.actor.load_state_dict(torch.load(actor_filepath))
    #     self.policy.critic.load_state_dict(torch.load(critic_filepath))

    def write_reward(self, r, r2):

        self.writer.add_scalar('Test cumulative reward current mazes', r, self.log_idx)
        self.writer.add_scalar('Test final reward current mazes', r2, self.log_idx)
        self.log_idx += 1

    def push_batchdata(self, st, a, logprob, r, done):
        # adds a row of trajectory data to self.batchdata
        self.batchdata.states.append(st)
        self.batchdata.actions.append(a)
        self.batchdata.logprobs.append(logprob)
        self.batchdata.rewards.append(r)
        self.batchdata.is_terminal.append(done)

    def clear_batchdata(self):
        self.batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(self.device)
        else:
            return torch.tensor(array, dtype=torch.float).to(self.device)


"""

REINFORCE agent

"""
class REINFORCE(nn.Module):
    def __init__(self, params, logdir, device):
        super(REINFORCE, self).__init__()

        self.batchdata = BatchData()
        self.writer = SummaryWriter(log_dir=logdir)
        self.log_idx = 0
        self.device = device

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        lr = 0.001
        self.gamma = 0.9
        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_decay = 0.98

        self.episode_per_update = params['episode_per_update']

        self.policy = Actor(10)
        # if load_pretrained:  # if load actor-critic network params from file
        #     self.load_model()
        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.policy_optim = RMSprop(self.policy.parameters(), lr=lr)

    def get_action(self, state, test=False):
        a, log_prob = self.policy.get_action(self.to_tensor(state))
        if np.random.random() > self.epsilon or test:
            return a, log_prob
        else:
            a = np.random.randint(0, 4)
            logprob = self.policy.evaluate(self.to_tensor(state), a)
            return a, logprob

    def push_data(self, s, a, logprob, r, done, s1):
        self.push_batchdata(s, a, logprob, r, done)

    def update(self, epoch):

        if epoch % self.episode_per_update == self.episode_per_update-1:
            self.update_Pi()
            self.clear_batchdata()  # reset the sampled policy trajectories
            self.epsilon *= self.epsilon_decay

    def update_Pi(self):

        rtgs = self.to_tensor(calc_rtg(self.batchdata.rewards, self.batchdata.is_terminal, self.gamma))
        # Normalize rewards
        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-5)
        logprobs = torch.cat([x.view(1) for x in self.batchdata.logprobs])

        loss = - rtgs*logprobs #+ self.c*(logprobs - np.log(1./self.action_dim))**2

        self.policy_optim.zero_grad()
        loss.mean().backward()
        self.policy_optim.step()

    # def save_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):  # TODO filename param
    #     torch.save(self.policy.actor.state_dict(), actor_filepath)
    #     torch.save(self.policy.critic.state_dict(), critic_filepath)
    #
    # def load_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):
    #     self.policy.actor.load_state_dict(torch.load(actor_filepath))
    #     self.policy.critic.load_state_dict(torch.load(critic_filepath))

    def write_reward(self, r, r2):

        self.writer.add_scalar('Test cumulative reward current mazes', r, self.log_idx)
        self.writer.add_scalar('Test final reward current mazes', r2, self.log_idx)
        self.log_idx += 1

    def push_batchdata(self, st, a, logprob, r, done):
        self.batchdata.states.append(st)
        self.batchdata.actions.append(a)
        self.batchdata.logprobs.append(logprob)
        self.batchdata.rewards.append(r)
        self.batchdata.is_terminal.append(done)

    def clear_batchdata(self):
        self.batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(self.device)
        else:
            return torch.tensor(array, dtype=torch.float).to(self.device)

