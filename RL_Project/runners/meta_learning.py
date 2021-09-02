import numpy as np
from tqdm import tqdm
import torch
import time

from RL_Project.agents.meta_ppo_agent import PPO
from RL_Project.environment import Simulator
# from agents.meta_ppo_agent import PPO
# from environment import Simulator


ADAPTATION = 0
EVALUATION = 1


def push_trajectories(agent, idx, sim, N_trj, T, theta_i=None, mode=ADAPTATION):

    if mode == ADAPTATION:
        agent.epsilon = agent.epsilon_adapt
    else:
        agent.epsilon = agent.epsilon0

    cumulative_rwds = 0
    num_optim_trj = 0

    for trj in range(N_trj):
        sim.reset()
        st = sim.get_state()
        temp_r = 0
        t_temp = 0

        for t in range(T):
            a, logprob, v = agent.get_action(st, theta_i)
            r, done = sim.step(a)
            temp_r += r
            t_temp += 1

            st1 = sim.get_state()

            if done:
                num_optim_trj += 1

            if t == T - 1:
                done = True

            agent.push_batchdata(st, a, logprob, v, r, done, mode, idx)

            if done:
                break

            st = st1

        cumulative_rwds += temp_r / t_temp

        if mode == EVALUATION:
            agent.epsilon *= agent.epsilon_decay

    return cumulative_rwds / N_trj, num_optim_trj


def simulate_trj(sim, agent, theta, path_len, test=True):
    tot_reward = 0
    r = 0
    sim.reset()
    st = sim.get_state()
    for t in range(path_len):
        a, _, _ = agent.get_action(st, theta=theta, test=test)
        r, done = sim.step(a)
        st1 = sim.get_state()
        tot_reward += r
        if done:
            break
        st = st1
    return tot_reward, r


def meta_learning(params, logdir, device):

    mazes = None
    paths_length = []

    agent = PPO(params, logdir, device, adaptive_lr=params['adaptive_lr']).to(device)

    # Epoch is number of batches where we want to train
    for epoch in tqdm(range(params['epochs'])):

        agent.clear_batchdata()

        ''' OLD POLICIES AND TRAJECTORY COLLECTION'''

        # collect trajectories and old policies for a batch and define the batches...
        avg_mean_rwd_trj = 0
        avg_num_opt_trj = 0
        avg_loss = 0
        avg_loss_pi = 0
        avg_loss_v = 0
        avg_R = 0
        avg_final_r = 0

        sims = []

        for i in range(params['batch_tasks']):

            sim = Simulator(params)
            sims.append(sim)
            grid = np.expand_dims(sim.grid.copy(), 0)
            paths_length.append(sim.T)
            if mazes is None:
                mazes = grid
            else:
                mazes = np.concatenate((mazes, grid), 0)

            T = sim.T * params['horizon_multiplier']

            # sample traj adaptation for each maze
            mean_rwd_trj, num_opt_trj = push_trajectories(agent, i, sim, params['adaptation_trajectories'], T, theta_i=None, mode=ADAPTATION)

            avg_mean_rwd_trj += mean_rwd_trj
            avg_num_opt_trj += num_opt_trj

            # adapt theta for each maze
            theta_i, loss_adapt, l_pi, l_v = agent.adapt(i, train=True, print_grads=True)
            avg_loss += loss_adapt
            avg_loss_pi += l_pi
            avg_loss_v += l_v

            # sample traj evaluation for each maze
            _, _ = push_trajectories(agent, i, sim, params['episodes'], T, theta_i=theta_i, mode=EVALUATION)

            R, final_r = simulate_trj(sim, agent, theta_i, sim.T, test=True)
            avg_R += R
            avg_final_r += final_r

        agent.writer.add_scalar("Adaptation loss", avg_loss/params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Adaptation policy loss", avg_loss_pi / params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Adaptation value loss", avg_loss_v / params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Mean reward adaptation", avg_mean_rwd_trj/params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Number optimal trajectories adaptation", avg_num_opt_trj/params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Test cumulative reward new mazes", avg_R/params['batch_tasks'], epoch)
        agent.writer.add_scalar("Test final reward new mazes", avg_final_r/params['batch_tasks'], epoch)
        for j, grad in enumerate(agent.grads_vals):
            agent.writer.add_scalar('params_grad_' + str(j), grad/params['batch_tasks'], agent.log_grads_idx)
        agent.grads_vals *= 0
        agent.log_grads_idx += 1

        ''' K PPO UPDATES OF THETA-0'''

        for k in range(params['episode_per_update']):
            l_tot = 0
            for i in range(params['batch_tasks']):

                # adapt theta-0
                theta_i, loss_adapt, _, _ = agent.adapt(i, train=True)

                # compute PPO loss
                l_i = agent.get_loss(theta_i, i)
                l_tot += l_i

            # Update theta-0 with sum of losses
            agent.optimizer.zero_grad()
            l_tot.backward()
            if params['gradient_clipping'] == 1:
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            agent.optimizer.step()

            # Update logporbs and state values of the adaptation trajectories for the new theta zero
            agent.update_adaptation_batches()

        avg_R = 0
        avg_final_r = 0
        for i in range(params['batch_tasks']):
            # adapt theta-0
            theta_i, _, _, _ = agent.adapt(i, train=False)

            R, final_r = simulate_trj(sims[i], agent, theta_i, sims[i].T, test=True)
            avg_R += R
            avg_final_r += final_r

        agent.writer.add_scalar('Test cumulative reward current mazes', avg_R/params['batch_tasks'], epoch)
        agent.writer.add_scalar('Test final reward current mazes', avg_final_r/params['batch_tasks'], epoch)

        ''' TEST ON OLD MAZES '''

        if epoch > 0:
            avg_reward = 0
            avg_final_r = 0

            first_maze = max(0, epoch-20)
            diff_mazes = epoch - first_maze
            for i, temp_maze in enumerate(mazes[first_maze:epoch]):
                x = first_maze+i

                sim = Simulator(params, grid=mazes[x], T=paths_length[x])
                T = paths_length[x]

                agent.clear_batchdata()
                _, _ = push_trajectories(agent, 0, sim, params['adaptation_trajectories'], T, theta_i=None, mode=ADAPTATION)
                theta_i, _, _, _ = agent.adapt(0, train=False)

                tmp_reward, final_r = simulate_trj(sim, agent, theta_i, paths_length[x], test=True)
                avg_reward += tmp_reward
                avg_final_r += final_r

            agent.writer.add_scalar("Test cumulative reward past mazes", avg_reward / diff_mazes, int(epoch))
            agent.writer.add_scalar("Test final reward past mazes", avg_final_r / diff_mazes, int(epoch))

    print()



