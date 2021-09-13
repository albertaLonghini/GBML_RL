import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random
from networks import MamlParamsPg, MamlParamsPPO, Forward_Predictor, Reward_Predictor
from utils import BatchData, calc_rtg
import numpy as np
from torch.optim import Adam, RMSprop, SGD

"""

PPO meta-agent

"""

class PPO(nn.Module):
    def __init__(self, params, writer, device, adaptive_lr=False, load_pretrained=False):
        super(PPO, self).__init__()

        self.action_dim = 4
        self.show_goal = params['show_goal']
        self.batchdata = [[BatchData() for _ in range(params['batch_tasks'])], [BatchData() for _ in range(params['batch_tasks'])]]        # list[0] for adaptation batch and list[1] for evaluation batch
        # self.writer = SummaryWriter(log_dir=logdir)
        self.writer = writer
        self.log_idx = 0
        self.log_grads_idx = 0
        self.device = device

        self.log_frq_idx = 0

        self.inner_lr = params['inner_lr']

        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_adapt = 0.9
        self.epsilon_adapt_decay = params['eps_adapt_decay']
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

        self.add_l2 = params['add_loss_exploration']
        self.inner_loss_type = params['inner_loss_type']
        self.idx_reward_pred = 0

        self.decouple_models = params['decoupled_explorer']

        self.ADAPTATION = 0
        self.EVALUATION = 1

        self.norm_A = params['norm_A']

        self.grad_align = params['gradient_alignment']

        self.policy = MamlParamsPPO(params['grid_size'], self.show_goal, self.inner_lr, params['inner_loss_type'], params['decoupled_explorer'], activation=params['activation'], adaptive_lr=adaptive_lr)

        self.grads_vals = np.zeros(len(self.policy.get_theta()))

        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, eps=params['adam_epsilon'])

        if self.add_l2 == 1:
            self.reward_prediction_model = Reward_Predictor(params['grid_size'], self.show_goal)
            self.optimizer_curiosity = Adam(self.reward_prediction_model.parameters(), lr=params['curiosity_lr'])

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

    def adapt(self, idx, train=False, print_grads=False):

        self.policy.train()

        theta_i = self.policy.get_theta()

        if idx == 0 and print_grads:
            states = np.concatenate(self.batchdata[0][idx].states, 0)
            img = np.sum(states, 0)[1]
            self.writer.add_image("exploration_frq", torch.tensor(img), self.log_frq_idx, dataformats='HW')
            self.log_frq_idx += 1

        rt = self.to_tensor(self.batchdata[0][idx].rewards)
        states = torch.cat([self.to_tensor(x) for x in self.batchdata[0][idx].states], 0).detach()
        actions = self.to_tensor(self.batchdata[0][idx].actions).view(-1, 1).detach()
        # logprobs = torch.cat([x.view(1) for x in self.batchdata[0][idx].logprobs])
        old_logprobs = torch.cat([x.view(1) for x in self.batchdata[0][idx].old_logprobs])

        logprobs, v, _ = self.policy.evaluate(states, actions[:,0], theta=theta_i)

        if self.inner_loss_type == 0:

            rtgs = self.to_tensor(calc_rtg(rt, self.batchdata[0][idx].is_terminal, self.gamma))  # reward-to-go
            # Normalize rewards

            # v = torch.cat([x.view(1) for x in self.batchdata[0][idx].v])
            A = rtgs - v

            loss_pi = ( - torch.exp(logprobs - old_logprobs.detach()) * A.detach() ).mean()  # (- rtgs * logprobs).mean() # TODO: importance sampling / something to update psi
            loss_v = (self.c1*(v - rtgs)**2).mean()
            loss = loss_pi + loss_v
            loss_pi = loss_pi.detach().cpu().item()
            loss_v = loss_v.detach().cpu().item()

        elif self.inner_loss_type == 1:

            rt_hat = self.policy.get_predicted_reward(states, actions)
            # loss = ((rt - rt_hat)**2).mean()
            # TODO: doesn't work, psi doesn't get updated
            R = ((rt - rt_hat) ** 2)
            A = R #- v#.detach()
            loss_pi = (- torch.exp(logprobs - old_logprobs.detach()) * A).mean()
            # loss_v = ((R.detach()-v)**2).mean()
            # logprobs, _, _ = self.policy.evaluate(states, actions[:,0])
            loss = loss_pi #+ loss_v
            loss_pi = loss_pi.detach().cpu().item()
            loss_v = 0#loss_v.detach().cpu().item()

        if train:
            theta_grad_s = torch.autograd.grad(outputs=loss, inputs=theta_i, create_graph=True)
            theta_i = list(map(lambda p: p[1] - p[2] * p[0], zip(theta_grad_s, theta_i, self.policy.lr)))

            if print_grads:
                for i, grad in enumerate(theta_grad_s):
                    self.grads_vals[i] += torch.mean(torch.abs(grad))

        else:
            theta_grad_s = torch.autograd.grad(outputs=loss, inputs=theta_i)
            theta_i = list(map(lambda p: p[1] - p[2] * p[0].detach(), zip(theta_grad_s, theta_i, self.policy.lr)))

        return theta_i, loss.detach().cpu().item(), loss_pi, loss_v, theta_grad_s

    def update_adaptation_batches(self):

        for batchdata in (self.batchdata[0]):
            states = torch.cat([self.to_tensor(x) for x in batchdata.states], 0).detach()
            actions = self.to_tensor(batchdata.actions).long().detach()
            logprobs, state_vals, _ = self.policy.evaluate(states, actions)#, theta=self.policy.get_theta())

            batchdata.logprobs = [x.detach() for x in logprobs]
            batchdata.v = [x.detach() for x in state_vals]

    def get_loss(self, theta_i, theta_0, idx, grads_adapt=None):
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
        actor_loss = torch.mean(- torch.min(surr1, surr2) )  # minus to maximize

        # Critic loss fitting to reward-to-go with entropy bonus
        critic_loss = self.c1 * self.MSE_loss(rtgs, state_vals)

        loss = actor_loss + critic_loss - self.c2 * H.mean()

        if theta_0 is not None:
            log_pi_theta, _, _ = self.policy.evaluate(old_states, old_actions, theta=self.policy.get_theta())
            log_pi_theta_0, _, _ = self.policy.evaluate(old_states, old_actions, theta=theta_0)
            loss += self.nu * self.kl(log_pi_theta, log_pi_theta_0.detach())

        if self.add_l2 == 1:
            loss += self.get_l2(self.batchdata[0][idx])

        if self.grad_align == 1:
            grads_evaluate = torch.autograd.grad(outputs=(actor_loss + critic_loss), inputs=theta_i, retain_graph=True)
            cosine_sim = self.cos(torch.cat([x.view(-1) for x in grads_evaluate]), torch.cat([x.view(-1) for x in grads_adapt]))

            # loss -= 100. * gradient_loss

            return loss, cosine_sim.detach().cpu().item()


        return loss, 0


    def get_l2(self, D):

        rt = self.to_tensor(D.rewards)
        states = torch.cat([self.to_tensor(x) for x in D.states], 0).detach()
        next_states = torch.cat([self.to_tensor(x) for x in D.next_states], 0).detach()
        actions = self.to_tensor(D.actions).view(-1, 1).detach()
        # logprobs = torch.cat([x.view(1) for x in D.logprobs])

        logprobs, _, _ = self.policy.evaluate(states, actions[:,0])

        rt_hat = (rt - self.reward_prediction_model(states, actions))**2
        self.optimizer_curiosity.zero_grad()
        rt_hat.mean().backward()
        self.optimizer_curiosity.step()

        self.writer.add_scalar("Reward prediction loss", rt_hat.mean().detach().cpu().numpy(), self.idx_reward_pred)
        self.idx_reward_pred += 1

        rtgs = self.to_tensor(calc_rtg(rt_hat.detach().cpu().numpy(), D.is_terminal, self.gamma))  # reward-to-go

        return (- logprobs * rtgs.detach()).mean()

    def save_model(self, filepath='./ppo_model.pth'):  # TODO filename param
        torch.save(self.policy.state_dict(), filepath)

    def load_model(self, filepath='./ppo_model.pth'):
        self.policy.load_state_dict(torch.load(filepath))

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

    def push_batchdata(self, st, a, logprob, old_logprob, v, r, done, st1, mode, idx):
        # adds a row of trajectory data to self.batchdata
        self.batchdata[mode][idx].states.append(st)
        self.batchdata[mode][idx].actions.append(a)
        self.batchdata[mode][idx].logprobs.append(logprob.detach())
        self.batchdata[mode][idx].old_logprobs.append(old_logprob.detach())
        self.batchdata[mode][idx].v.append(v.detach())
        self.batchdata[mode][idx].rewards.append(r)
        self.batchdata[mode][idx].is_terminal.append(done)
        self.batchdata[mode][idx].next_states.append(st1)

    def clear_batchdata(self):
        for i in range(2):
            for batchdata in self.batchdata[i]:
                batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(self.device)
        else:
            return torch.tensor(array, dtype=torch.float).to(self.device)


"""

REINFORCE meta-agent

"""
class REINFORCE(nn.Module):
    def __init__(self, params, logdir, device, adaptive_lr=False, load_pretrained=False):
        super(REINFORCE, self).__init__()

        self.action_dim = 4
        self.show_goal = params['show_goal']
        self.batchdata = [[BatchData() for _ in range(params['batch_tasks'])], [BatchData() for _ in range(params['batch_tasks'])]]
        self.writer = SummaryWriter(log_dir=logdir)
        self.log_idx = 0
        self.log_grads_idx = 0
        self.device = device

        self.inner_lr = 0.1
        # self.training_steps = 10

        self.epsilon0 = 0.9
        self.epsilon = self.epsilon0
        self.epsilon_adapt = params['eps_adapt']
        self.epsilon_decay = 0.95

        self.K = 1

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr = 0.001 #0.001  # 0.01
        self.gamma = 0.9

        self.ADAPTATION = 0
        self.EVALUATION = 1

        self.policy = MamlParamsPg(params['grid_size'], self.show_goal, self.inner_lr, model_type=params['model_type'], activation=params['activation'], adaptive_lr=adaptive_lr)

        self.grads_vals = np.zeros(len(self.policy.get_theta()))

        self.MSE_loss = nn.MSELoss()  # to calculate critic loss
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)

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

    def get_action(self, state, theta=None, test=False):
        # Sample actions with epsilon greedy
        if np.random.random() > self.epsilon or test:
            a, log_prob, _ = self.policy(self.to_tensor(state), theta=theta)
            return a, log_prob, None
        else:
            a = np.random.randint(0, 3)
            log_prob, _, _ = self.policy.evaluate(self.to_tensor(state), a, theta=theta)
            return a, log_prob, None

    def adapt(self, idx, train=False, print_grads=False):

        self.policy.train()

        theta_i = self.policy.get_theta()

        rtgs = self.to_tensor(calc_rtg(self.batchdata[0][idx].rewards, self.batchdata[0][idx].is_terminal, self.gamma))  # reward-to-go
        # Normalize rewards
        # rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-5)  # todo: ?
        logprobs = torch.cat([x.view(1) for x in self.batchdata[0][idx].logprobs])

        # Normalize advantages
        # advantages = (A-A.mean()) / (A.std() + 1e-5)
        loss = (- rtgs * logprobs).mean()

        if train:
            theta_grad_s = torch.autograd.grad(outputs=loss, inputs=theta_i, create_graph=True)
            theta_i = list(map(lambda p: p[1] - p[2] * p[0], zip(theta_grad_s, theta_i, self.policy.lr)))

            if print_grads:
                for i, grad in enumerate(theta_grad_s):
                    self.grads_vals[i] += torch.mean(torch.abs(grad))
        else:
            theta_grad_s = torch.autograd.grad(outputs=loss, inputs=theta_i)
            theta_i = list(map(lambda p: p[1] - p[2] * p[0].detach(), zip(theta_grad_s, theta_i, self.policy.lr)))

        return theta_i, loss.detach().cpu().item(), 0, 0

    def update_adaptation_batches(self):

        for batchdata in (self.batchdata[0]):
            states = torch.cat([self.to_tensor(x) for x in batchdata.states], 0).detach()
            actions = self.to_tensor(batchdata.actions).long().detach()
            logprobs, state_vals, _ = self.policy.evaluate(states, actions)

            batchdata.logprobs = [x for x in logprobs]
            batchdata.v = [None]*len(logprobs)

    def get_loss(self, theta_i, idx):

        rtgs = self.to_tensor(calc_rtg(self.batchdata[1][idx].rewards, self.batchdata[1][idx].is_terminal, self.gamma))  # reward-to-go
        # Normalize rewards
        # rtgs = ((rtgs - torch.mean(rtgs)) / torch.std(rtgs)).detach()

        old_states = torch.cat([self.to_tensor(x) for x in self.batchdata[1][idx].states], 0).detach()
        old_actions = self.to_tensor(self.batchdata[1][idx].actions).long().detach()

        # get form correct batch new policy data
        logprobs, _, _ = self.policy.evaluate(old_states, old_actions, theta=theta_i)

        loss = - rtgs*logprobs

        return loss.mean()

    # def regularize(self):
    #     actions = self.batchdata.actions
    #     states = torch.cat([self.to_tensor(x) for x in self.batchdata.states], 0)
    #     _, entropy = self.policy.evaluate(states, actions)
    #     loss = - entropy
    #
    #     return loss


    def save_model(self, filepath='./reinforce_model.pth'):  # TODO filename param
        torch.save(self.policy.state_dict(), filepath)

    def load_model(self, filepath='./reinforce_model.pth'):
        self.policy.load_state_dict(torch.load(filepath))

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

    def push_batchdata(self, st, a, logprob, v, r, done, st1, mode, idx):
        # adds a row of trajectory data to self.batchdata
        self.batchdata[mode][idx].states.append(st)
        self.batchdata[mode][idx].actions.append(a)
        self.batchdata[mode][idx].logprobs.append(logprob)
        self.batchdata[mode][idx].v.append(v)
        self.batchdata[mode][idx].rewards.append(r)
        self.batchdata[mode][idx].is_terminal.append(done)
        self.batchdata[mode][idx].next_states.append(st1)

    def clear_batchdata(self):
        for i in range(2):
            for batchdata in self.batchdata[i]:
                batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(self.device)
        else:
            return torch.tensor(array, dtype=torch.float).to(self.device)