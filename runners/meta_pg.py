import numpy as np
from env import Simulator
from agents.pg_meta_agent import REINFORCE, PPO
from maze_gen import Maze_Gen
from tqdm import tqdm
import torch
import time
import copy

from filtering_methods import random_filter, max_reward, state_frequencies, forward_prediction, max_inner_loss


def push_trajectories(agent, idx, sim, N_trj, ratio_best_trj, T, theta_i=None, mode=0, optim_traj=None, optimal=False, filter=0):

    agent.epsilon = agent.get_epsilon(mode)

    temp_data = []
    cumulative_rwds = np.zeros(max(1, N_trj))
    temp_num_optim_trj = np.zeros(max(1, N_trj))

    '''
    optimal trajectory given by oracle
    '''
    if optimal:
        optimal_path = optim_traj[np.random.randint(0, len(optim_traj))]
        sim.reset()
        st = sim.get_state()
        for a in optimal_path:
            r0, done = sim.step(a)
            log_prob, v, _ = agent.policy.evaluate(agent.to_tensor(st), a)

            st1 = sim.get_state()

            agent.push_batchdata(st, a, log_prob, log_prob, v, r0, done, st1, mode, idx)

            cumulative_rwds[0] += r0
            if done:
                temp_num_optim_trj[0] += 1

            st = st1

        return cumulative_rwds[0], temp_num_optim_trj[0]

    '''
    explore trajectories
    '''
    if N_trj == 0:
        sim.reset()
        s0 = sim.get_state()
        a0, logprob, v0 = agent.get_action(s0, theta_i)
        r0, done = sim.step(a0)
        st1 = sim.get_state()
        agent.push_batchdata(s0, a0, logprob, logprob, v0, r0, done, st1, mode, idx)

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
                # logprob, v, _ = agent.policy.evaluate(agent.to_tensor(st), a, theta=agent.policy.get_theta())
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
            agent.push_batchdata(batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5], batch_data[6], batch_data[7], mode, idx)

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


def meta_pg(params, writer, n_experiment, device):

    mazes = None
    paths_length = []
    maze_gen = Maze_Gen(params)
    # agent = PPO(params, logdir, device, adaptive_lr=params['adaptive_lr']).to(device)
    if params['agent'] == 'reinforce':
        agent = REINFORCE(params, writer, device, adaptive_lr=params['adaptive_lr']).to(device)
    if params['agent'] == 'ppo':
        agent = PPO(params, writer, device, adaptive_lr=params['adaptive_lr']).to(device)

    saved_distances = []

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

        avg_final_distance = 0

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

            sim = Simulator(start, goal, maze, path_len, params)
            T_adapt = path_len * params['horizon_multiplier_adaptation']
            T = path_len * params['horizon_multiplier']

            # sample traj adaptation for each maze
            mean_rwd_trj, num_opt_trj = push_trajectories(agent, i, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T_adapt, theta_i=None, mode=agent.ADAPTATION, optim_traj=paths, optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])
            avg_mean_rwd_trj += mean_rwd_trj
            avg_num_opt_trj += num_opt_trj

            # adapt theta for each maze
            theta_i, loss_adapt, l_pi, l_v, grads_adapt = agent.adapt(i, train=True, print_grads=True)
            avg_loss += loss_adapt
            avg_loss_pi += l_pi
            avg_loss_v += l_v

            # sample traj evaluation for each maze
            _, _ = push_trajectories(agent, i, sim, params['episodes'], 1, T, theta_i=theta_i, mode=agent.EVALUATION, optim_traj=paths, optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])

            R, final_r, final_distance = simulate_trj(sim, agent, theta_i, path_len, test=True)
            avg_R += sim.normalize_reward(R, path_len)
            avg_final_r += final_r

            avg_final_distance += final_distance

        saved_distances.append(avg_final_distance/params['batch_tasks'])

        if epoch % 10 == 0 and n_experiment != "":
            name = 'l2='+str(params['add_loss_exploration'])+'_l_inner='+str(params['inner_loss_type'])+'_decouple='+str(params['decoupled_explorer'])+'_R_sparse='+str(params['sparse_reward'])
            np.save("./results"+n_experiment+"/distances_"+name+".npy", np.array(saved_distances))

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


        # print(agent.policy.psi[-5], agent.policy.psi[-1])
        # print(agent.policy.get_theta()[-5])

        cos_tot = 0
        l_expl = 0

        for theta_it in range(params['exploiter_iteration']):
            # recollect trajectories with adapted theta, reset theta_0
            theta_0 = copy.deepcopy(agent.policy.get_theta())

            print("Iteration{}: ".format(theta_it),agent.policy.psi[-1], agent.policy.theta[-5], agent.policy.z_0[-1])

            if theta_it > 0:
                agent.clear_batchdata(1)

            for k in range(agent.K):

                print(torch.cuda.max_memory_allocated(device=device))

                l_tot = 0
                l2_tot = 0
                for i in range(params['batch_tasks']):

                    # adapt theta-0
                    theta_i, loss_adapt, _, _, grads_adapt = agent.adapt(i, train=True)

                    #recollect evaluation trajs
                    if theta_it > 0 and k == 0:
                        id_sim = params['batch_tasks']*epoch + i
                        sim = Simulator(starts[id_sim], goals[id_sim], mazes[id_sim], paths_length[id_sim], params)
                        T_sim = paths_length[id_sim] * params['horizon_multiplier']
                        _, _ = push_trajectories(agent, i, sim, params['episodes'], 1, T_sim, theta_i=theta_i, mode=agent.EVALUATION, optim_traj=None, optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])

                    # compute PPO loss
                    l_i, l2_i, cos_sim = agent.get_loss(theta_i, theta_0, i, grads_adapt)
                    l_tot += l_i
                    l2_tot += l2_i
                    cos_tot += cos_sim

                # l_expl = l_tot + l2_tot

                if agent.dec_opt == 1:
                    # Update theta-0 with sum of losses
                    agent.optimizer_exploiter.zero_grad()
                    l_tot.backward()
                    if params['gradient_clipping'] == 1:
                        torch.nn.utils.clip_grad_norm_(agent.model_parameters, 0.5)
                    agent.optimizer_exploiter.step()

                    # if last update of the agent then update explorer
                    if k == agent.K - 1 and theta_it == params['exploiter_iteration'] - 1:
                        agent.optimizer_explorer.zero_grad()
                        l2_tot.backward()
                        if params['gradient_clipping'] == 1:
                            torch.nn.utils.clip_grad_norm_(agent.explorer_parameters, 0.5)
                        agent.optimizer_explorer.step()
                else:
                    # Update theta-0 with sum of losses
                    if agent.decouple_models == 0:
                        agent.optimizer.zero_grad()
                        (l_tot + l2_tot).backward()
                        if params['gradient_clipping'] == 1:
                            torch.nn.utils.clip_grad_norm_(agent.model_parameters, 0.5)
                        agent.optimizer.step()
                    else:
                        # Update theta-0 with sum of losses
                        agent.optimizer_exploiter.zero_grad()
                        agent.optimizer_explorer.zero_grad()  # todo: check if there is a gradient
                        (l_tot + l2_tot).backward()
                        if params['gradient_clipping'] == 1:
                            torch.nn.utils.clip_grad_norm_(agent.model_parameters, 0.5)
                        agent.optimizer_exploiter.step()

                        # if last update of the agent then update explorer
                        if k == agent.K - 1 and theta_it == params['exploiter_iteration'] - 1:

                            if params['gradient_clipping'] == 1:
                                torch.nn.utils.clip_grad_norm_(agent.explorer_parameters, 0.5)
                            agent.optimizer_explorer.step()



            # # Update logporbs and state values of the adaptation trajectories for the new theta zero
            # agent.update_adaptation_batches()

        agent.writer.add_scalar("Cosine similarity", cos_tot / (params['batch_tasks'] * agent.K), int(epoch))

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
        #         T_adapt = int(paths_length[x] * params['horizon_multiplier_adaptation'])
        #         T = int(paths_length[x] * params['horizon_multiplier'])
        #
        #         agent.clear_batchdata()
        #         _, _ = push_trajectories(agent, 0, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T_adapt, theta_i=None, mode=agent.ADAPTATION, optim_traj=old_paths[x], optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])
        #         theta_i, _, _, _, _ = agent.adapt(0, train=False)
        #
        #         tmp_reward, final_r = simulate_trj(sim, agent, theta_i, T, test=True)
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
        # T_adapt = rnd_paths_length * params['horizon_multiplier_adaptation']
        # T = rnd_paths_length * params['horizon_multiplier']
        #
        # agent.clear_batchdata()
        # _, _ = push_trajectories(agent, 0, rnd_sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], T_adapt, theta_i=None, mode=agent.ADAPTATION, optim_traj=rnd_paths, optimal=params['adaptation_optimal_traj'], filter=params['filter_type'])
        # theta_i, _, _, _, _ = agent.adapt(0, train=False)
        #
        # tot_reward, final_r = simulate_trj(rnd_sim, agent, theta_i, T, test=True)
        # tot_reward = rnd_sim.normalize_reward(tot_reward, T)
        #
        # agent.writer.add_scalar("Test reward on random starting point", tot_reward, int(epoch))
        # agent.writer.add_scalar("Test final reward on random starting point", final_r, int(epoch))


    print()



