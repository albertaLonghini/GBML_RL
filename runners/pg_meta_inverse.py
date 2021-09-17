import numpy as np
from env import Simulator
from agents.pg_meta_inverse_agent import PPO, Meta_PPO
from maze_gen import Maze_Gen
from tqdm import tqdm
import torch
from torch import nn
import time
import glob
import matplotlib.pyplot as plt

from filtering_methods import random_filter, max_reward, state_frequencies, forward_prediction, max_inner_loss


def push_trajectories(agent, sim, N_trj, ratio_best_trj, T, theta_i=None, mode=0, optim_traj=None, optimal=False, filter=0):

    agent.epsilon = agent.get_epsilon(mode)

    temp_data = []
    cumulative_rwds = np.zeros(max(1, N_trj))
    temp_num_optim_trj = np.zeros(max(1, N_trj))

    '''
    optimal trajectory given by oracle
    '''
    # if optimal:
    #     optimal_path = optim_traj[np.random.randint(0, len(optim_traj))]
    #     sim.reset()
    #     st = sim.get_state()
    #     for a in optimal_path:
    #         r0, done = sim.step(a)
    #         log_prob, v, _ = agent.policy.evaluate(agent.to_tensor(st), a)
    #
    #         st1 = sim.get_state()
    #
    #         agent.push_batchdata(st, a, log_prob, v, r0, done, st1, mode, idx)
    #
    #         cumulative_rwds[0] += r0
    #         if done:
    #             temp_num_optim_trj[0] += 1
    #
    #         st = st1
    #
    #     return cumulative_rwds[0], temp_num_optim_trj[0]

    '''
    explore trajectories
    '''
    # if N_trj == 0:
    #     sim.reset()
    #     s0 = sim.get_state()
    #     a0, logprob, v0 = agent.get_action(s0, theta_i)
    #     r0, done = sim.step(a0)
    #     st1 = sim.get_state()
    #     agent.push_batchdata(s0, a0, logprob, v0, r0, done, st1, mode, idx)
    #
    #     cumulative_rwds[0] += r0
    #     if done:
    #         temp_num_optim_trj[0] += 1
    #
    # else:
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

            temp_traj.append((st, a, logprob, logprob, v, r, done, st1))

            if done:
                break

            st = st1

        temp_data.append(temp_traj)

        agent.update_epsilon(mode)

    if mode == agent.ADAPTATION:
        if filter == 0:
            indeces = random_filter(N_trj, ratio_best_trj)
        elif filter == 1:
            indeces = max_reward(cumulative_rwds)
        elif filter == 2:
            indeces = state_frequencies(temp_data, use_actions=False)
        elif filter == 3:
            indeces = state_frequencies(temp_data, use_actions=True)
        elif filter == 4:
            indeces = forward_prediction(agent, temp_data)
        elif filter == 5:
            indeces = max_inner_loss(agent, temp_data)

        N_best_trj = max(1, int(N_trj * ratio_best_trj))
    else:
        indeces = np.array(range(N_trj))
        N_best_trj = N_trj

    num_optim_trj = 0
    tot_rwd_trj = 0

    for i in list(indeces[N_trj - N_best_trj:]):
        num_optim_trj += temp_num_optim_trj[i]
        tot_rwd_trj += cumulative_rwds[i]
        for batch_data in temp_data[i]:
            agent.push_batchdata(batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5], batch_data[6], batch_data[7])

    return tot_rwd_trj / N_best_trj, num_optim_trj


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
    return tot_reward, r, sim.get_distance()


def inverse_meta_pg(params, logdir, device, writer):

    starting_theta_stars = []
    starting_starts = []
    starting_goals = []
    starting_mazes = []
    starting_path_lens = []
    for file in glob.glob("./saved_policies/*/*"):
        numpy_file = np.load(file)
        starting_theta_stars.append(numpy_file['arr_0'])  # np.reshape(numpy_file['arr_0'], (1, -1))
        starting_starts.append(numpy_file['arr_1'])
        starting_goals.append(numpy_file['arr_2'])
        starting_mazes.append(numpy_file['arr_3'])
        starting_path_lens.append(numpy_file['arr_4'])

    mazes = None
    paths_length = []
    maze_gen = Maze_Gen(params)
    meta_agent = Meta_PPO(params, device, adaptive_lr=params['adaptive_lr']).to(device)



    L_tot = 0
    counter = 0
    for epoch in range(1000):


        for i in tqdm(range(len(starting_theta_stars))):
            meta_agent.clear_batchdata()
            sim = Simulator(tuple(starting_starts[i]), tuple(starting_goals[i]), starting_mazes[i], int(starting_path_lens[i]), params)
            T_adapt = starting_path_lens[i] * params['horizon_multiplier_adaptation']
            T = starting_path_lens[i] * params['horizon_multiplier']

            # sample traj adaptation for each maze
            _, _ = push_trajectories(meta_agent, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T_adapt, theta_i=None, mode=meta_agent.ADAPTATION, filter=params['filter_type'])

            # adapt theta for each maze
            theta_i, _, _, _, img = meta_agent.adapt(train=True, print_grads=False)

            tot_reward, final_r, distance = simulate_trj(sim, meta_agent, theta_i, T, test=True)

            theta_i_flat = torch.cat([t.view(-1) for t in theta_i])
            theta_star_flat = torch.tensor(starting_theta_stars[i]).to(device).detach()

            L = torch.mean((theta_i_flat - theta_star_flat)**2)
            L_tot += L
            counter += 1

            writer.add_scalar("distance", distance, counter)
            writer.add_scalar("loss", L.detach().cpu().item(), counter)


            if counter % 100 == 99:
                meta_agent.optimizer.zero_grad()
                L_tot.backward()
                meta_agent.optimizer.step()
                L_tot = 0

                writer.add_image("exploration_frq", torch.tensor(img), counter, dataformats='HW')


    print()





    L_tot = 0

    theta_stars = []

    # Epoch is number of batches where we want to train
    for epoch in range(params['epochs']):

        meta_agent.clear_batchdata()

        ''' OLD POLICIES AND TRAJECTORY COLLECTION'''

        # # collect trajectories and old policies for a batch and define the batches...
        # avg_mean_rwd_trj = 0
        # avg_num_opt_trj = 0
        # avg_loss = 0
        # avg_loss_pi = 0
        # avg_loss_v = 0
        # avg_R = 0
        # avg_final_r = 0


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

        sim = Simulator(start, goal, maze, path_len, params)
        T_adapt = path_len * params['horizon_multiplier_adaptation']
        T = path_len * params['horizon_multiplier']

        # sample traj adaptation for each maze
        _, _ = push_trajectories(meta_agent, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T_adapt, theta_i=None, mode=meta_agent.ADAPTATION, optim_traj=paths, optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])

        # adapt theta for each maze
        theta_i, _, _, _ = meta_agent.adapt(train=True, print_grads=False)

        tot_reward, r = simulate_trj(sim, meta_agent, theta_i, path_len, test=True)
        # meta_agent.writer.add_scalar("Test tot reward", tot_reward, epoch)
        writer.add_scalar("Test final reward", r, epoch)

        """
            For each task train optimal policy starting from theta_i
        """
        #DummyMAML(params['grid_size'], params['show_goal'], theta_i, model_type=params['model_type']).state_dict()
        agent = PPO(params, logdir, device, theta_i).to(device)
        agent.clear_batchdata()
        solved_to_go = 100
        for episode in tqdm(range(params['episodes'])):
            sim.reset()
            st = sim.get_state()
            for t in range(T):
                a, logprob = agent.get_action(st)
                r, done = sim.step(a)
                st1 = sim.get_state()
                if t == T - 1:
                    done = True
                agent.push_batchdata(st, a, logprob, r, done)
                if done:
                    break
                st = st1
            if r == 1:
                solved_to_go -= 1
            else:
                solved_to_go = 100
            writer.add_scalar('Training final reward', r, episode+epoch*params['episodes'])
            if solved_to_go <= 0:
                break
            if episode % params['episode_per_update'] == params['episode_per_update']-1:
                agent.update()
                agent.clear_batchdata()

        """
            here we assume policy has trained to the optimum and adapt theta_0
        """
        theta_stars.append([theta.detach() for theta in agent.policy.theta_0])

        if epoch % 10 == 9:
            for rgs_idx in range(params['regression_steps']):
                L_tot = 0
                avg_mean_rwd_trj = 0
                avg_num_opt_trj = 0
                for j, theta_star in enumerate(theta_stars):
                    meta_agent.clear_batchdata()

                    sim_j = Simulator(starts[j], goals[j], mazes[j], paths_length[j], params)

                    # sample traj adaptation for each maze
                    mean_rwd_trj, num_opt_trj = push_trajectories(meta_agent, sim_j, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T_adapt, theta_i=None, mode=meta_agent.ADAPTATION, optim_traj=paths, optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])
                    avg_mean_rwd_trj += mean_rwd_trj
                    avg_num_opt_trj += num_opt_trj

                    # adapt theta for each maze
                    theta_i, loss_adapt, l_pi, l_v = meta_agent.adapt(train=True, print_grads=True)
                    # avg_loss += loss_adapt
                    # avg_loss_pi += l_pi
                    # avg_loss_v += l_v

                    L = torch.mean(torch.cat([((t1 - t2)**2).view(-1) for t1, t2 in zip(theta_i, theta_star)]))
                    L_tot += L

                meta_agent.optimizer.zero_grad()
                L_tot.backward()
                # if params['gradient_clipping'] == 1:
                #     torch.nn.utils.clip_grad_norm_(meta_agent.policy.parameters(), 0.5)
                meta_agent.optimizer.step()

                writer.add_scalar("Mean reward adaptation", avg_mean_rwd_trj / len(theta_stars), epoch * params['regression_steps'] + rgs_idx)
                writer.add_scalar("Number optimal trajectories adaptation", avg_num_opt_trj / len(theta_stars), epoch * params['regression_steps'] + rgs_idx)
                writer.add_scalar("Loss regression", L_tot.detach().cpu().item() / len(theta_stars), epoch * params['regression_steps'] + rgs_idx)
                for j, grad in enumerate(meta_agent.grads_vals):
                    writer.add_scalar('params_grad_' + str(j), grad/len(theta_stars), meta_agent.log_grads_idx)
                meta_agent.grads_vals *= 0
                meta_agent.log_grads_idx += 1





            # # sample traj evaluation for each maze
            # _, _ = push_trajectories(agent, i, sim, params['episodes'], 1, T, theta_i=theta_i, mode=agent.EVALUATION, optim_traj=paths, optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])
            #
            # R, final_r = simulate_trj(sim, agent, theta_i, path_len, test=True)
            # avg_R += sim.normalize_reward(R, path_len)
            # avg_final_r += final_r

        # meta_agent.writer.add_scalar("Adaptation loss", avg_loss/params['batch_tasks'], int(epoch))
        # meta_agent.writer.add_scalar("Adaptation policy loss", avg_loss_pi / params['batch_tasks'], int(epoch))
        # meta_agent.writer.add_scalar("Adaptation value loss", avg_loss_v / params['batch_tasks'], int(epoch))
        # meta_agent.writer.add_scalar("Mean reward adaptation", avg_mean_rwd_trj/params['batch_tasks'], int(epoch))
        # meta_agent.writer.add_scalar("Number optimal trajectories adaptation", avg_num_opt_trj/params['batch_tasks'], int(epoch))
        # # agent.writer.add_scalar("Test reward theta prime before training", avg_R/params['batch_tasks'], int(epoch))
        # # agent.writer.add_scalar("Final test reward theta prime before training", avg_final_r/params['batch_tasks'], int(epoch))
        # for j, grad in enumerate(meta_agent.grads_vals):
        #     meta_agent.writer.add_scalar('params_grad_' + str(j), grad/params['batch_tasks'], meta_agent.log_grads_idx)
        # meta_agent.grads_vals *= 0
        # meta_agent.log_grads_idx += 1





















        # ''' TEST ON OLD MAZES '''
        #
        # if epoch > 0:
        #     avg_reward = 0
        #     avg_final_r = 0
        #
        #     first_maze = max(0, epoch-20)
        #     diff_mazes = epoch - first_maze
        #     for i, temp_maze in enumerate(mazes[first_maze:epoch]):
        #         x = first_maze+i
        #
        #         sim = Simulator(starts[x], goals[x], temp_maze, paths_length[x], params)
        #         T = int(paths_length[x] * params['horizon_multiplier'])
        #
        #         agent.clear_batchdata()
        #         _, _ = push_trajectories(agent, 0, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T, theta_i=None, mode=agent.ADAPTATION, optim_traj=old_paths[x], optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])
        #         theta_i, _, _, _ = agent.adapt(0, train=False)
        #
        #         tmp_reward, final_r = simulate_trj(sim, agent, theta_i, paths_length[x], test=True)
        #         avg_reward += sim.normalize_reward(tmp_reward, paths_length[x])
        #         avg_final_r += final_r
        #
        #     agent.writer.add_scalar("Previous mazes average reward", avg_reward / diff_mazes, int(epoch))
        #     agent.writer.add_scalar("Previous mazes final reward", avg_final_r / diff_mazes, int(epoch))
        #
        # ''' TEST IN RANDOM STARTING POINT AND MAZE'''
        #
        # rnd_start, rnd_goal, rnd_maze, rnd_paths_length, rnd_paths = maze_gen.get_maze(central=False)
        # rnd_sim = Simulator(rnd_start, rnd_goal, rnd_maze, rnd_paths_length, params)
        # T = rnd_paths_length * params['horizon_multiplier']
        #
        # agent.clear_batchdata()
        # _, _ = push_trajectories(agent, 0, rnd_sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T, theta_i=None, mode=agent.ADAPTATION, optim_traj=rnd_paths, optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])
        # theta_i, _, _, _ = agent.adapt(0, train=False)
        #
        # tot_reward, final_r = simulate_trj(rnd_sim, agent, theta_i, rnd_paths_length, test=True)
        # tot_reward = rnd_sim.normalize_reward(tot_reward, rnd_paths_length)
        #
        # agent.writer.add_scalar("Test reward on random starting point", tot_reward, int(epoch))
        # agent.writer.add_scalar("Test final reward on random starting point", final_r, int(epoch))


    print()



