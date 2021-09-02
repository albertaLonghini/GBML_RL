import numpy as np
from env import Simulator
from agents.pg_meta_agent import REINFORCE, PPO
from maze_gen import Maze_Gen
from tqdm import tqdm
import torch
import time

from filtering_methods import max_reward


def push_trajectories(agent, idx, sim, N_trj, ratio_best_trj, T, theta_i=None, mode=0, optim_traj=None, optimal=False):

    agent.epsilon = agent.get_epsilon(mode)

    temp_data = []
    cumulative_rwds = np.zeros(max(1, N_trj))
    temp_num_optim_trj = np.zeros(max(1, N_trj))

    if optimal:
        optimal_path = optim_traj[np.random.randint(0, len(optim_traj))]
        sim.reset()
        st = sim.get_state()
        for a in optimal_path:
            r0, done = sim.step(a)
            log_prob, v, _ = agent.policy.evaluate(agent.to_tensor(st), a)
            agent.push_batchdata(st, a, log_prob, v, r0, done, mode, idx)

            st1 = sim.get_state()

            cumulative_rwds[0] += r0
            if done:
                temp_num_optim_trj[0] += 1

            st = st1

        return cumulative_rwds[0], temp_num_optim_trj[0]
    else:
        if N_trj == 0:
            sim.reset()
            s0 = sim.get_state()
            a0, logprob, v0 = agent.get_action(s0, theta_i)
            r0, done = sim.step(a0)
            agent.push_batchdata(s0, a0, logprob, v0, r0, done, mode, idx)

            cumulative_rwds[0] += r0
            if done:
                temp_num_optim_trj[0] += 1

        else:
            for trj in range(N_trj):
                sim.reset()
                st = sim.get_state()
                temp_traj = []

                for t in range(T):
                    a, logprob, v = agent.get_action(st, theta_i)
                    r, done = sim.step(a)
                    cumulative_rwds[trj] += r

                    st1 = sim.get_state()

                    if done:
                        temp_num_optim_trj[trj] += 1

                    if t == T - 1:
                        done = True

                    temp_traj.append((st, a, logprob, v, r, done))

                    if done:
                        break

                    st = st1

                temp_data.append(temp_traj)

                agent.update_epsilon(mode)

    return max_reward(agent, temp_data, mode, idx, N_trj, ratio_best_trj, cumulative_rwds, temp_num_optim_trj)


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


def meta_PPO(params, logdir, device):

    mazes = None
    paths_length = []
    maze_gen = Maze_Gen(params)
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

        for i in range(params['batch_tasks']):

            start, goal, maze, path_len, paths = maze_gen.get_maze()
            paths_length.append(path_len)
            if mazes is None:
                mazes = np.expand_dims(maze, 0)
                starts = [start]
                goals = [goal]
                old_paths = [paths]
            else:
                mazes = np.concatenate((mazes, np.expand_dims(maze, 0)), 0)
                starts.append(start)
                goals.append(goal)
                old_paths.append(paths)

            sim = Simulator(start, goal, maze, params)
            T = path_len * params['horizon_multiplier']

            # sample traj adaptation for each maze
            mean_rwd_trj, num_opt_trj = push_trajectories(agent, i, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T, theta_i=None, mode=agent.ADAPTATION, optim_traj=paths, optimal=params['adaptation_optimal_traj'])
            avg_mean_rwd_trj += mean_rwd_trj
            avg_num_opt_trj += num_opt_trj

            # adapt theta for each maze
            theta_i, loss_adapt, l_pi, l_v = agent.adapt(i, train=True, print_grads=True)
            avg_loss += loss_adapt
            avg_loss_pi += l_pi
            avg_loss_v += l_v

            # sample traj evaluation for each maze
            _, _ = push_trajectories(agent, i, sim, params['episodes'], 1, T, theta_i=theta_i, mode=agent.EVALUATION, optim_traj=paths, optimal=params['adaptation_optimal_traj'])

            R, final_r = simulate_trj(sim, agent, theta_i, path_len, test=True)
            avg_R += sim.normalize_reward(R, params['path_length'])
            avg_final_r += final_r

        agent.writer.add_scalar("Adaptation loss", avg_loss/params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Adaptation policy loss", avg_loss_pi / params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Adaptation value loss", avg_loss_v / params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Mean reward adaptation", avg_mean_rwd_trj/params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Number optimal trajectories adaptation", avg_num_opt_trj/params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Test reward theta prime before training", avg_R/params['batch_tasks'], int(epoch))
        agent.writer.add_scalar("Final test reward theta prime before training", avg_final_r/params['batch_tasks'], int(epoch))
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

        ''' TEST ON OLD MAZES '''

        if epoch > 0:
            avg_reward = 0
            avg_final_r = 0

            first_maze = max(0, epoch-20)
            diff_mazes = epoch - first_maze
            for i, temp_maze in enumerate(mazes[first_maze:epoch]):
                x = first_maze+i

                sim = Simulator(starts[x], goals[x], temp_maze, params)
                T = int(paths_length[x] * params['horizon_multiplier'])

                agent.clear_batchdata()
                _, _ = push_trajectories(agent, 0, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T, theta_i=None, mode=agent.ADAPTATION, optim_traj=old_paths[x], optimal=params['adaptation_optimal_traj'])
                theta_i, _, _, _ = agent.adapt(0, train=False)

                tmp_reward, final_r = simulate_trj(sim, agent, theta_i, paths_length[x], test=True)
                avg_reward += sim.normalize_reward(tmp_reward, params['path_length'])
                avg_final_r += final_r

            agent.writer.add_scalar("Previous mazes average reward", avg_reward / diff_mazes, int(epoch))
            agent.writer.add_scalar("Previous mazes final reward", avg_final_r / diff_mazes, int(epoch))

        ''' TEST IN RANDOM STARTING POINT AND MAZE'''

        rnd_start, rnd_goal, rnd_maze, rnd_paths_length, rnd_paths = maze_gen.get_maze(central=False)
        rnd_sim = Simulator(rnd_start, rnd_goal, rnd_maze, params)
        T = rnd_paths_length * params['horizon_multiplier']

        agent.clear_batchdata()
        _, _ = push_trajectories(agent, 0, rnd_sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T, theta_i=None, mode=agent.ADAPTATION, optim_traj=rnd_paths, optimal=params['adaptation_optimal_traj'])
        theta_i, _, _, _ = agent.adapt(0, train=False)

        tot_reward, final_r = simulate_trj(rnd_sim, agent, theta_i, rnd_paths_length, test=True)
        tot_reward = rnd_sim.normalize_reward(tot_reward, params['path_length'])

        agent.writer.add_scalar("Test reward on random starting point", tot_reward, int(epoch))
        agent.writer.add_scalar("Test final reward on random starting point", final_r, int(epoch))


    print()



