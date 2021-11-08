import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random
from networks import MamlParamsPg, NormalMamlParamsPPO, Curiosity1MamlParamsPPO, Curiosity2MamlParamsPPO, Forward_Predictor, Reward_Predictor
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
        self.epsilon_decay = 0.6 #0.85

        self.K = params['episode_per_update']

        # Init params and actor-critic policy networks, old_policy used for sampling, policy for training
        self.lr = params['lr']
        self.eps_clip = params['eps_clip']
        self.gamma = 0.9
        self.c1 = params['c1']
        self.c2 = params['c2']

        # Parameters for additive loss
        self.reg_l2 = params['reg_l2']
        self.cl2 = params['cl2']

        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.nu = params['kl_nu']

        self.add_l2 = params['add_loss_exploration']
        self.inner_loss_type = params['inner_loss_type']
        self.idx_reward_pred = 0
        self.dec_opt = params['decoupled_optimization']
        self.explorer_loss = params['explorer_loss']
        self.decouple_models = params['decoupled_explorer']
        self.decouple_predictors = params['decoupled_predictors']
        self.beta_model = params['beta_model']

        self.ADAPTATION = 0
        self.EVALUATION = 1

        self.norm_A = params['norm_A']

        self.grad_align = params['gradient_alignment']
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        self.MSE_loss = nn.MSELoss()  # to calculate critic loss

        if self.inner_loss_type == 0:
            self.policy = NormalMamlParamsPPO(params, activation=params['activation'], adaptive_lr=adaptive_lr)
            self.model_parameters = self.policy.parameters()
            self.main_optimizer = Adam(self.model_parameters, lr=self.lr, eps=params['adam_epsilon'])
        else:
            if self.beta_model == 0:
                self.policy = Curiosity1MamlParamsPPO(params, activation=params['activation'], adaptive_lr=adaptive_lr)
            else:
                self.policy = Curiosity2MamlParamsPPO(params, activation=params['activation'], adaptive_lr=adaptive_lr)

            if self.decouple_models == 1:
                self.psi_params = [param for param in self.policy.psi.parameters()]
                self.optimizer_explorer = Adam(self.psi_params, lr=self.lr, eps=params['adam_epsilon'])

            body_params = self.policy.params_to_adapt
            beta_params = [param for param in self.policy.beta.parameters()]
            v_params = [param for param in self.policy.theta_v.parameters()]
            if adaptive_lr:
                lr_params = [param for param in self.policy.lr.parameters()]
            else:
                lr_params = []

            if self.decouple_predictors == 1:
                self.model_parameters = body_params + beta_params + v_params + lr_params
            else:
                self.model_parameters = body_params + v_params + lr_params
                predictors_parameters = beta_params
                self.optimizer_predictor = Adam(predictors_parameters, lr=self.lr, eps=params['adam_epsilon'])

            # if self.dec_opt == 1:
            #     self.optimizer_explorer = Adam(psi_params, lr=self.lr, eps=params['adam_epsilon'])
            # else:
            #     self.model_parameters += psi_params

            self.main_optimizer = Adam(self.model_parameters, lr=self.lr, eps=params['adam_epsilon'])





        self.grads_vals = np.zeros(len(self.policy.get_exploiter_starting_params()))

        # if self.dec_opt == 0 and self.decouple_models == 0:  # TODO: optimizer_predictors in case decouple_predictors==1 even for this case?
        #     if self.decouple_predictors == 1:
        #         self.model_parameters = self.policy.parameters()
        #         self.optimizer = Adam(self.model_parameters, lr=self.lr, eps=params['adam_epsilon'])
        #     else:
        #
        # else:
        #
        #     psi_params = [param for param in self.policy.psi.parameters()]
        #     body_params = self.policy.params_to_adapt
        #     beta_params = [param for param in self.policy.beta.parameters()]
        #     v_params = [param for param in self.policy.theta_v.parameters()]
        #     if adaptive_lr:
        #         lr_params = [param for param in self.policy.lr.parameters()]
        #     else:
        #         lr_params = []
        #
        #     if self.decouple_predictors == 1:
        #         self.exploitation_parameters = body_params + beta_params + v_params + lr_params
        #     else:
        #         self.exploitation_parameters = body_params + v_params + lr_params
        #         self.predictors_parameters = beta_params
        #         self.optimizer_predictors = Adam(self.predictors_parameters, lr=self.lr, eps=params['adam_epsilon'])
        #
        #     self.explorer_parameters = psi_params
        #     self.optimizer_exploiter = Adam(self.model_parameters, lr=self.lr, eps=params['adam_epsilon'])
        #     self.optimizer_explorer = Adam(psi_params, lr=self.lr, eps=params['adam_epsilon'])

        if self.add_l2 == 1:
            self.reward_prediction_model = Reward_Predictor(params['grid_size'], self.show_goal)
            self.optimizer_curiosity = Adam(self.reward_prediction_model.parameters(), lr=params['curiosity_lr'])

        if params['filter_type'] == 4:
            self.forward_model = Forward_Predictor(params['grid_size'], self.show_goal)
            self.optimizer_forward = Adam(self.forward_model.parameters(), lr=0.0001)               # TODO: maybe try different values of lr

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

        rt = self.to_tensor(self.batchdata[0][idx].rewards)
        rtgs = self.to_tensor(calc_rtg(rt, self.batchdata[0][idx].is_terminal, self.gamma))  # reward-to-go
        states = torch.cat([self.to_tensor(x) for x in self.batchdata[0][idx].states], 0).detach()
        actions = self.to_tensor(self.batchdata[0][idx].actions).view(-1, 1).detach()
        old_logprobs_explorer = torch.cat([x.view(1) for x in self.batchdata[0][idx].old_logprobs])
        terminals = self.batchdata[0][idx].is_terminal

        logprobs_explorer, v, _ = self.policy.evaluate(states, actions[:, 0])  # logprobs of explorer

        loss, loss_pi, loss_v, R_err, rt_hat = self.policy.get_adapt_loss(logprobs_explorer, old_logprobs_explorer, rt, rtgs, states, actions, v, self.c1, terminals, self.device)



        if idx == 0 and print_grads:
            img = np.zeros((17, 17))
            for idx_img, states_img in enumerate(self.batchdata[0][0].states):
                t = idx_img % (int(states_img.shape[1] / 2) - 1)
                img[int(states_img[0, 2 * t]), int(states_img[0, 2 * t + 1])] += 1

                # for t in range(int(states.shape[1] / 2)):
                #     img[int(states[0, 2 * t]), int(states[0, 2 * t + 1])] += 1
            self.writer.add_image("exploration_visited", torch.tensor(img), self.log_frq_idx, dataformats='HW')
            self.log_frq_idx += 1

            # img = np.zeros((17, 17))
            # counter = np.ones((17, 17))
            # for idx_img, states_img in enumerate(self.batchdata[0][0].next_states):
            #
            #     t = idx_img % (int(states_img.shape[1] / 2) - 1)
            #     img[int(states_img[0, 2 * t]), int(states_img[0, 2 * t + 1])] += rt_hat[idx_img]
            #     counter[int(states_img[0, 2 * t]), int(states_img[0, 2 * t + 1])] += 1
            #
            #
            #     # for t in range(int(states.shape[1] / 2)):
            #     #     img[int(states[0, 2 * t]), int(states[0, 2 * t + 1])] += 1
            #     #     counter[int(states[0, 2 * t]), int(states[0, 2 * t + 1])] += 1
            # img /= counter
            # self.writer.add_image("inner_reward", torch.tensor(img), self.log_frq_idx, dataformats='HW')
            # self.log_frq_idx += 1



        # if self.decouple_predictors == 0 and self.inner_loss_type == 1:
        #     rt_hat = self.policy.get_predicted_reward(states, actions, use_beta=True)
        #     R_err_beta = ((rt - rt_hat) ** 2)
        #
        #     self.optimizer_predictor.zero_grad()
        #     R_err_beta.mean().backward()
        #     self.optimizer_predictor.step()
        #
        #     self.writer.add_scalar("Reward prediction loss", R_err_beta.mean().detach().cpu().numpy(), self.idx_reward_pred)
        #     self.idx_reward_pred += 1





        # theta_i = self.policy.get_theta()       # if self.beta_model == 1, this term is a list = [z_0, theta]
        #
        #
        #
        #
        # rt = self.to_tensor(self.batchdata[0][idx].rewards)
        # states = torch.cat([self.to_tensor(x) for x in self.batchdata[0][idx].states], 0).detach()
        # actions = self.to_tensor(self.batchdata[0][idx].actions).view(-1, 1).detach()
        # # logprobs = torch.cat([x.view(1) for x in self.batchdata[0][idx].logprobs])
        # old_logprobs = torch.cat([x.view(1) for x in self.batchdata[0][idx].old_logprobs])
        #
        # if self.decouple_models == 1:
        #     logprobs, v, _ = self.policy.evaluate(states, actions[:, 0])
        # else:
        #     logprobs, v, _ = self.policy.evaluate(states, actions[:, 0], theta=theta_i)
        #
        # inner_pred_MSE= 0
        # if self.inner_loss_type == 0:
        #
        #     rtgs = self.to_tensor(calc_rtg(rt, self.batchdata[0][idx].is_terminal, self.gamma))  # reward-to-go
        #     # Normalize rewards
        #
        #     # v = torch.cat([x.view(1) for x in self.batchdata[0][idx].v])
        #     A = rtgs - v
        #
        #     loss_pi = ( - torch.exp(logprobs - old_logprobs.detach()) * A.detach()).mean()  # (- rtgs * logprobs).mean() # TODO: importance sampling / something to update psi
        #     loss_v = (self.c1*(v - rtgs)**2).mean()
        #     loss = loss_pi + loss_v
        #     loss_pi = loss_pi.detach().cpu().item()
        #     loss_v = loss_v.detach().cpu().item()
        #
        # elif self.inner_loss_type == 1:
        #
        #     rt_hat = self.policy.get_predicted_reward(states, actions)
        #     # loss = ((rt - rt_hat)**2).mean()
        #     R = ((rt - rt_hat) ** 2)
        #
        #     # train beta model with MSE
        #
        #     inner_pred_MSE = R.mean().detach().cpu().item()
        #     A = R #- v#.detach()
        #     loss_pi = (- torch.exp(logprobs - old_logprobs.detach()) * A).mean()
        #     # loss_v = ((R.detach()-v)**2).mean()
        #     # logprobs, _, _ = self.policy.evaluate(states, actions[:,0])
        #     loss = loss_pi #+ loss_v
        #     loss_pi = loss_pi.detach().cpu().item()
        #     loss_v = 0#loss_v.detach().cpu().item()

        if train:
            theta_grad_s = torch.autograd.grad(outputs=loss, inputs=self.policy.get_exploiter_starting_params(), create_graph=True)
            theta_i = list(map(lambda p: p[1] - p[2] * p[0], zip(theta_grad_s, self.policy.get_exploiter_starting_params(), self.policy.lr)))

            if print_grads:
                for i, grad in enumerate(theta_grad_s):
                    self.grads_vals[i] += torch.mean(torch.abs(grad)).detach()

            # if self.decouple_predictors == 0:
            #     self.optimizer_predictor.zero_grad()
            #     R_err.backward(retain_graph=True)
            #     self.optimizer_predictor.step()

        else:
            theta_grad_s = torch.autograd.grad(outputs=loss, inputs=self.policy.get_exploiter_starting_params())
            theta_i = list(map(lambda p: p[1] - p[2] * p[0].detach(), zip(theta_grad_s, self.policy.get_exploiter_starting_params(), self.policy.lr)))

        return theta_i, loss.detach().cpu().item(), loss_pi, loss_v, theta_grad_s, R_err

    # def update_adaptation_batches(self):
    #
    #     for batchdata in (self.batchdata[0]):
    #         states = torch.cat([self.to_tensor(x) for x in batchdata.states], 0).detach()
    #         actions = self.to_tensor(batchdata.actions).long().detach()
    #         logprobs, state_vals, _ = self.policy.evaluate(states, actions)#, theta=self.policy.get_theta())
    #
    #         batchdata.logprobs = [x.detach() for x in logprobs]
    #         batchdata.v = [x.detach() for x in state_vals]

    def get_loss(self, theta_i, theta_0, idx, last_iteration=False, grads_adapt=None):
        # get form correct batch old policy data
        rtgs = self.to_tensor(calc_rtg(self.batchdata[1][idx].rewards, self.batchdata[1][idx].is_terminal, self.gamma))  # reward-to-go

        old_states = torch.cat([self.to_tensor(x) for x in self.batchdata[1][idx].states], 0).detach()
        old_actions = self.to_tensor(self.batchdata[1][idx].actions).long().detach()
        old_logprobs_exploiter = torch.cat([x.view(1) for x in self.batchdata[1][idx].logprobs]).detach()

        #get form correct batch new policy data
        logprobs_exploiter, state_vals, H = self.policy.evaluate(old_states, old_actions, theta=theta_i)

        # Compute loss
        # Importance ratio
        ratios = torch.exp(logprobs_exploiter - old_logprobs_exploiter.detach())  # new probs over old probs

        # Calc advantages
        A = rtgs - state_vals
        if self.norm_A == 1:
            A = ((A - torch.mean(A)) / torch.std(A)).detach()

        # Actor loss using CLIP loss
        surr1 = ratios * A
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * A
        actor_loss = torch.mean(- torch.min(surr1, surr2))  # minus to maximize

        # Critic loss fitting to reward-to-go with entropy bonus
        critic_loss = self.c1 * self.MSE_loss(rtgs, state_vals)

        loss = actor_loss + critic_loss - self.c2 * H.mean()

        if theta_0 is not None:  # TODO: can theta_0 be None
            log_pi_theta, _, _ = self.policy.evaluate(old_states, old_actions, theta=self.policy.get_exploiter_starting_params())
            log_pi_theta_0, _, _ = self.policy.evaluate(old_states, old_actions, theta=theta_0)
            loss += self.nu * self.kl(log_pi_theta, log_pi_theta_0.detach())

        loss_2 = 0
        if self.add_l2 == 1 and last_iteration:
            loss_2 = self.get_l2(self.batchdata[0][idx])

        if self.grad_align == 1:
            grads_evaluate = torch.autograd.grad(outputs=actor_loss, inputs=theta_i, retain_graph=True)
            cosine_sim = self.cos(torch.cat([x.view(-1) for x in grads_evaluate]), torch.cat([x.view(-1) for x in grads_adapt]))

            # loss -= 100. * gradient_loss

            return loss, loss_2, cosine_sim.detach().cpu().item()

        return loss, loss_2, 0


    def get_l2(self, D):

        rt = self.to_tensor(D.rewards)
        states = torch.cat([self.to_tensor(x) for x in D.states], 0).detach()
        next_states = torch.cat([self.to_tensor(x) for x in D.next_states], 0).detach()
        actions = self.to_tensor(D.actions).view(-1, 1).detach()

        logprobs_explorer, _, dist_entropy = self.policy.evaluate(states, actions[:, 0])

        if self.decouple_predictors == 1:
            rt_pred = self.reward_prediction_model(states, actions)
            rt_hat = (rt - rt_pred) ** 2

            self.optimizer_curiosity.zero_grad()
            rt_hat.mean().backward()
            self.optimizer_curiosity.step()

            self.writer.add_scalar("Reward prediction loss", rt_hat.mean().detach().cpu().numpy(), self.idx_reward_pred)
            self.idx_reward_pred += 1
        else:
            rt_pred = self.policy.get_predicted_reward(states, actions)
            rt_hat = (rt - rt_pred) ** 2

        # rt_hat_var = ((rt_hat - rt_hat.mean()) ** 2).mean()  # todo: variance of the rewards
        # rt_hat_mean = rt_hat.mean()
        # if self.reg_l2 ==1:
        #     rt_hat_mean += rt_hat_var

        # unique_states = []
        # zero_out = []
        # for state, terminal in zip(next_states, D.is_terminal):
        #     if terminal:
        #         unique_states = []
        #     if any([(state == x).all() for x in unique_states]):
        #         zero_out.append(0)
        #     else:
        #         unique_states.append(state)
        #         zero_out.append(1)

        if self.explorer_loss == 0:
            rt_hat_zero_out = rt_hat.detach().cpu().numpy()# * np.array(zero_out)
            rtgs = self.to_tensor(calc_rtg(rt_hat_zero_out, D.is_terminal, self.gamma))  # reward-to-go £
            return self.cl2*(- logprobs_explorer * rtgs.detach()).mean()

        if self.explorer_loss == 1:
            entropy_coefficient = 0.01
            return self.cl2*(-entropy_coefficient*dist_entropy.mean())

        if self.explorer_loss == 2:
            entropy_coefficient = 0.001
            rt_hat_zero_out = rt_hat.detach().cpu().numpy()  # * np.array(zero_out)
            rtgs = self.to_tensor(calc_rtg(rt_hat_zero_out, D.is_terminal, self.gamma))  # reward-to-go £
            return self.cl2 * ((- logprobs_explorer * rtgs.detach()).mean() - entropy_coefficient*dist_entropy.mean())

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
        self.batchdata[mode][idx].v.append(v if isinstance(v, int) else v.detach())
        self.batchdata[mode][idx].rewards.append(r)
        self.batchdata[mode][idx].is_terminal.append(done)
        self.batchdata[mode][idx].next_states.append(st1)

    def clear_batchdata(self, idx=None):
        if idx != None:
            for batchdata in self.batchdata[idx]:
                batchdata.clear()
        else:
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

    def clear_batchdata(self, idx=None):
        for i in range(2):
            for batchdata in self.batchdata[i]:
                batchdata.clear()

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).float().to(self.device)
        else:
            return torch.tensor(array, dtype=torch.float).to(self.device)