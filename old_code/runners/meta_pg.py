import numpy as np
from env import Simulator
from agents.pg_meta_agent import REINFORCE, PPO
from maze_gen import Maze_Gen
from tqdm import tqdm
import torch


def get_adapted_theta(agent, sim, N_trj, ratio_best_trj, T, train=False, optim_traj=None, optimal=False):
    agent.clear_batchdata()
    agent.epsilon = agent.epsilon_adapt

    temp_data = []
    cumulative_rwds = np.zeros(max(1, N_trj))
    temp_num_optim_trj = np.zeros(max(1, N_trj))

    if optimal:
        optimal_path = optim_traj[np.random.randint(0,len(optim_traj))]
        sim.reset()
        st = sim.get_state()
        for a in optimal_path:
            r0, done = sim.step(a)
            log_prob, v = agent.policy.evaluate(agent.to_tensor(st), a)
            agent.push_batchdata(st, a, log_prob, v, r0, done)

            st1 = sim.get_state()

            cumulative_rwds[0] += r0
            if done:
                temp_num_optim_trj[0] += 1

            st = st1

        theta_i, loss_adapt = agent.adapt(train=train)
        agent.clear_batchdata()
        return theta_i, loss_adapt, cumulative_rwds[0], temp_num_optim_trj[0]
    else:
        if N_trj == 0:
            sim.reset()
            s0 = sim.get_state()
            a0, logprob, v0 = agent.get_action(s0)
            r0, done = sim.step(a0)
            agent.push_batchdata(s0, a0, logprob, v0, r0, done)

            cumulative_rwds[0] += r0
            if done:
                temp_num_optim_trj[0] += 1

        else:
            for trj in range(N_trj):
                sim.reset()
                st = sim.get_state()
                temp_traj = []

                for t in range(T):
                    a, logprob, v = agent.get_action(st)
                    r, done = sim.step(a)
                    cumulative_rwds[trj] += r

                    st1 = sim.get_state()

                    if done:
                        temp_num_optim_trj[trj] += 1

                    if t == T - 1:
                        done = True

                    # agent.push_batchdata(st, a, logprob, r, done)
                    temp_traj.append((st, a, logprob, v, r, done))

                    if done:
                        break

                    st = st1

                temp_data.append(temp_traj)

    indeces = np.argsort(cumulative_rwds)           # todo: different ways of choosing indeces

    N_best_trj = max(1, int(N_trj * ratio_best_trj))
    num_optim_trj = 0
    tot_rwd_trj = 0

    for i in list(indeces[N_trj - N_best_trj:]):
        num_optim_trj += temp_num_optim_trj[i]
        tot_rwd_trj += cumulative_rwds[i]
        for batch_data in temp_data[i]:
            agent.push_batchdata(batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5])


    theta_i, loss_adapt = agent.adapt(train=train)
    agent.clear_batchdata()
    return theta_i, loss_adapt, tot_rwd_trj/N_best_trj, num_optim_trj


def meta_pg(params, logdir, device):

    mazes = None
    paths_length = []
    maze_gen = Maze_Gen(params)

    if params['agent'] == 'reinforce':
        agent = REINFORCE(params, logdir, device).to(device)
    if params['agent'] == 'ppo':
        agent = PPO(params, logdir, device).to(device)

    l_tot = 0

    # train over multiple MDPs batches
    for epoch in tqdm(range(params['epochs'])):

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
        # frqs = np.zeros((params['grid_size']+2, params['grid_size']+2))

        ''' ADAPTATION STEP '''

        theta_i, loss_adapt, mean_rwd_trj, num_opt_trj = get_adapted_theta(agent, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'], path_len*params['horizon_multiplier_adaptation'], train=True, optim_traj=paths, optimal=params['adaptation_optimal_traj']) #agent.adapt(train=True)
        agent.writer.add_scalar("Adaptation loss", loss_adapt, int(epoch))
        agent.writer.add_scalar("Mean reward adaptation", mean_rwd_trj, int(epoch))
        agent.writer.add_scalar("Number optimal trajectories adaptation", num_opt_trj, int(epoch))


        ''' TEST THETA PRIME ON NEW MAZE '''

        tot_reward = 0
        sim.reset()
        st = sim.get_state()
        for t in range(path_len):
            a, _, _ = agent.get_action(st, theta=theta_i, test=True)
            r, done = sim.step(a)
            st1 = sim.get_state()
            tot_reward += r
            if done:
                break
            st = st1
        agent.writer.add_scalar("Test reward before training", sim.normalize_reward(tot_reward, params['path_length']), int(epoch))

        ''' EVALUATION STEP '''
        agent.clear_batchdata()

        avg_prob = 0
        actual_t = 0
        agent.epsilon = agent.epsilon0              # only if agent uses epsilon greedy

        for e in range(params['episodes']):

            tot_reward = 0
            final_r = 0

            sim.reset()
            st = sim.get_state()

            for t in range(T):
                # frqs[0, sim.actual_pos_x, sim.actual_pos_y] += 1
                a, logprob, v = agent.get_action(st, theta=theta_i)
                r, done = sim.step(a)

                avg_prob += torch.exp(logprob).detach().cpu().item()
                actual_t += 1

                st1 = sim.get_state()

                # frqs += st1[0, 1]

                tot_reward += r
                final_r = r

                if t == T - 1:
                    done = True

                agent.push_batchdata(st, a, logprob, v, r, done)

                if done:
                    break

                st = st1

            # TODO: decrease epsilon for multiple trajectories evaluations ???
            agent.epsilon *= agent.epsilon_decay

        agent.writer.add_scalar("Average taken action probability", avg_prob/actual_t, int(epoch))
        tot_reward = sim.normalize_reward(tot_reward, params['path_length'])
        agent.write_reward(tot_reward, final_r)

        # Update the networks
        l_i = agent.update(theta_i)
        l_tot += l_i

        if epoch % params['batch_tasks'] == params['batch_tasks'] - 1:
            # backprop from the mean of the TD losses in the batch
            agent.optimizer.zero_grad()
            l_tot.backward()
            agent.optimizer.step()
            l_tot = 0

            ''' TEST NEW POLICY ON CURRENT MAZE '''
            theta_i, _, _, _ = get_adapted_theta(agent, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'],  path_len*params['horizon_multiplier_adaptation'], train=False, optim_traj=paths, optimal=params['adaptation_optimal_traj'])  #agent.adapt(train=False)

            tot_reward = 0
            sim.reset()
            st = sim.get_state()
            for t in range(path_len):
                a, _, _ = agent.get_action(st, theta=theta_i, test=True)
                r, done = sim.step(a)
                st1 = sim.get_state()
                tot_reward += r
                if done:
                    break
                st = st1
            agent.writer.add_scalar("Test reward", sim.normalize_reward(tot_reward, params['path_length']), int(epoch/params['batch_tasks']))

        ''' TEST ON OLD MAZES '''

        # Once trained in a new maze, test the performances in the previous mazes.
        if epoch > 0:
            tot_reward = 0
            first_maze = max(0, epoch-20)           # todo: sure about this thing?
            diff_mazes = epoch - first_maze
            for i, temp_maze in enumerate(mazes[first_maze:epoch]):
                x = first_maze+i

                sim = Simulator(starts[x], goals[x], temp_maze, params)
                T = int(paths_length[x] * params['horizon_multiplier'])

                theta_i, _, _, _ = get_adapted_theta(agent, sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'],  paths_length[x]*params['horizon_multiplier_adaptation'], train=False, optim_traj=old_paths[x], optimal=params['adaptation_optimal_traj'])  #agent.adapt(train=False)

                sim.reset()
                st = sim.get_state()
                tmp_reward = 0

                for t in range(paths_length[x]):

                    a, _, _ = agent.get_action(st, theta=theta_i, test=True)
                    r, done = sim.step(a)

                    st1 = sim.get_state()
                    tmp_reward += r

                    if done:
                        break
                    st = st1

                tmp_reward = sim.normalize_reward(tmp_reward, params['path_length'])
                tot_reward += tmp_reward

            agent.writer.add_scalar("Previous mazes average reward", tot_reward / diff_mazes, int(epoch))

        ''' TEST IN RANDOM STARTING POINT AND MAZE'''

        rnd_start, rnd_goal, rnd_maze, rnd_paths_length, rnd_paths = maze_gen.get_maze(central=False)
        rnd_sim = Simulator(rnd_start, rnd_goal, rnd_maze, params)
        T = rnd_paths_length * params['horizon_multiplier']


        theta_i, _, _, _ = get_adapted_theta(agent, rnd_sim, params['adaptation_trajectories'], params['adaptation_best_trajectories'],  rnd_paths_length*params['horizon_multiplier_adaptation'], train=False, optim_traj=rnd_paths, optimal=params['adaptation_optimal_traj'])  #agent.adapt(train=False)

        tot_reward = 0

        rnd_sim.reset()
        st = rnd_sim.get_state()

        for t in range(rnd_paths_length):
            a, _, _ = agent.get_action(st, theta=theta_i, test=True)
            r, done = rnd_sim.step(a)

            st1 = rnd_sim.get_state()

            tot_reward += r

            if done:
                break

            st = st1

        tot_reward = rnd_sim.normalize_reward(tot_reward, params['path_length'])
        agent.writer.add_scalar("Test reward on random starting point", tot_reward, int(epoch))


    print()
