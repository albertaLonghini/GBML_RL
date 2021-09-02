import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from networks import ActorCritic, Actor
from torch.optim import Adam, RMSprop


"""
Implements PPO-Clip-esk:
https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
or something along those lines
"""


class BatchData:  # batchdata collected from policy
    def __init__(self):
        self.states = []
        self.actions = []
        self.v = []
        self.logprobs = []  # log probs of each action
        self.rewards = []
        #self.lens = []  # episodic lengths in batch, (dim=n_episodes)
        self.is_terminal = []  # whether or not terminal state was reached

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.v.clear()
        self.rewards.clear()
        #self.lens.clear()
        self.is_terminal.clear()


def calc_rtg(rewards, is_terminals, gamma):
    # Calculates reward-to-go
    assert len(rewards) == len(is_terminals)
    rtgs = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + gamma * discounted_reward
        rtgs.insert(0, discounted_reward)
    return rtgs


"""

PPO agent

"""


class PPO:
    def __init__(self, params, logdir, device, load_pretrained=False):
        # extract environment info from maze....
        # self.mazesim = mazesim
        self.state_dim = 3  # I guess for 1 grid image?
        self.action_dim = 4  # {0: Down, 1: Up, 2: Right, 3: Left}
        self.batchdata = BatchData()
        self.writer = SummaryWriter(log_dir=logdir)
        self.log_idx = 0
        self.device = device

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr = 0.001
        self.eps_clip = 0.1
        self.gamma = 0.9
        self.c1 = 1.  # VF loss coefficient
        self.c2 = 0.1  # Entropy bonus coefficient
        self.K_epochs = 5  # num epochs to train on batch data
        self.epsilon = 0.9

        self.policy = ActorCritic(params['grid_size'], self.state_dim, self.action_dim).to(device)
        if load_pretrained:  # if load actor-critic network params from file
            self.load_model()
        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.policy_optim = RMSprop(self.policy.parameters(), self.lr)
        # self.policy_optim = Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': self.lr},
        #     {'params': self.policy.critic.parameters(), 'lr': self.lr}
        # ])

        self.old_policy = ActorCritic(params['grid_size'], self.state_dim, self.action_dim).to(device)
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

    def update(self):
        """
            Updates the actor-critic networks for current batch data
        """
        rtgs = self.to_tensor(calc_rtg(self.batchdata.rewards,self.batchdata.is_terminal,self.gamma))  # reward-to-go
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
            self.policy_optim.step()

        # Replace old policy with new policy
        self.old_policy.load_state_dict(self.policy.state_dict())

    def save_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):  # TODO filename param
        torch.save(self.policy.actor.state_dict(), actor_filepath)
        torch.save(self.policy.critic.state_dict(), critic_filepath)

    def load_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):
        self.policy.actor.load_state_dict(torch.load(actor_filepath))
        self.policy.critic.load_state_dict(torch.load(critic_filepath))

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

    # def set_mazesim(self, mazesim):
    #     # Assumes size of states and action space is the same as the previous one
    #     assert isinstance(mazesim, Simulator)
    #     self.mazesim = mazesim


"""

REINFORCE agent

"""


class REINFORCE:
    def __init__(self, params, logdir, device, load_pretrained=False):
        # extract environment info from maze....
        self.state_dim = 3  # I guess for 1 grid image?
        self.action_dim = 4  # {0: Down, 1: Up, 2: Right, 3: Left}
        self.batchdata = BatchData()
        self.writer = SummaryWriter(log_dir=logdir)
        self.log_idx = 0
        self.device = device

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr = 0.01
        self.gamma = 0.9
        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_decay = 0.997
        self.c = params['entropy_bonus']

        self.policy = Actor(params['grid_size'], self.state_dim, self.action_dim).to(device)
        if load_pretrained:  # if load actor-critic network params from file
            self.load_model()
        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.policy_optim = Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr}])

    def get_action(self, state, test=False):
        # Sample actions with epsilon greedy
        a, log_prob = self.policy.get_action(self.to_tensor(state))
        if np.random.random() > self.epsilon or test:
            return a, log_prob
        else:
            a = np.random.randint(0, 4)
            logprob = self.policy.evaluate(self.to_tensor(state), a)
            return a, logprob

    def update(self):
        """
            Updates the actor-critic networks for current batch data
        """
        rtgs = self.to_tensor(calc_rtg(self.batchdata.rewards, self.batchdata.is_terminal, self.gamma))  # reward-to-go
        # Normalize rewards
        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-5)
        # states = torch.cat([self.to_tensor(x) for x in self.batchdata.states], 0)
        # actions = self.to_tensor(self.batchdata.actions)
        # logprobs = self.policy.evaluate(states, actions.long())
        logprobs = torch.cat([x.view(1) for x in self.batchdata.logprobs])

        # Normalize advantages
        # advantages = (A-A.mean()) / (A.std() + 1e-5)
        loss = - rtgs*logprobs + self.c*(logprobs - np.log(1./self.action_dim))**2

        self.policy_optim.zero_grad()
        loss.mean().backward()
        self.policy_optim.step()
        self.epsilon *= self.epsilon_decay

    def save_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):  # TODO filename param
        torch.save(self.policy.actor.state_dict(), actor_filepath)
        torch.save(self.policy.critic.state_dict(), critic_filepath)

    def load_model(self, actor_filepath='./ppo_actor.pth', critic_filepath='./ppo_critic.pth'):
        self.policy.actor.load_state_dict(torch.load(actor_filepath))
        self.policy.critic.load_state_dict(torch.load(critic_filepath))

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

