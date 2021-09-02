import numpy as np
from env import Simulator
from agents.dqn_meta_agent import Net
from maze_gen import Maze_Gen
from tqdm import tqdm


def get_adapted_theta(agent, sim, train=False):
    sim.reset()
    s0 = sim.get_state()
    a0 = agent.get_action(s0)
    r0, done = sim.step(a0)
    s1 = sim.get_state()
    return agent.adapt(s0, a0, r0, (not done), s1, train=train)


def meta_dqn(params, logdir, device):

    mazes = None
    paths_length = []
    maze_gen = Maze_Gen(params)

    agent = Net(params, logdir, device).to(device)

    l_tot = 0

    # train over multiple MDPs batches
    for epoch in tqdm(range(params['epochs'])):

        start, goal, maze, path_len, _ = maze_gen.get_maze()
        paths_length.append(path_len)
        if mazes is None:
            mazes = np.expand_dims(maze, 0)
            starts = [start]
            goals = [goal]
        else:
            mazes = np.concatenate((mazes, np.expand_dims(maze, 0)), 0)
            starts.append(start)
            goals.append(goal)

        sim = Simulator(start, goal, maze, params)
        T = paths_length[-1] * params['horizon_multiplier']
        frqs = np.zeros((1, params['grid_size']+2, params['grid_size']+2))

        agent.epsilon = agent.epsilon0

        agent.D.reset()

        ''' ADAPTATION STEP '''

        # sim.reset()
        # s0 = sim.get_state()
        # a0 = agent.get_action(s0)
        # r0, done = sim.step(a0)
        # s1 = sim.get_state()
        theta_i = get_adapted_theta(agent, sim, train=True) #agent.adapt(s0, a0, r0, (not done), s1, train=True)

        ''' TEST THETA PRIME ON NEW MAZE '''

        tot_reward = 0
        sim.reset()
        st = sim.get_state()
        for t in range(path_len):
            a = agent.get_action(st, theta=theta_i, test=True)
            r, done = sim.step(a)
            st1 = sim.get_state()
            tot_reward += r
            if done:
                break
            st = st1
        tot_reward = sim.normalize_reward(tot_reward, params['path_length'])
        agent.writer.add_scalar("Test reward before training", tot_reward, int(epoch))

        ''' EVALUATION STEP '''

        for e in range(params['episodes']):

            tot_reward = 0
            final_r = 0

            sim.reset()
            st = sim.get_state()

            for t in range(T):
                frqs[0, sim.actual_pos_x, sim.actual_pos_y] += 1
                a = agent.get_action(st, theta=theta_i)
                r, done = sim.step(a)

                st1 = sim.get_state()

                tot_reward += r
                final_r = r

                if t == T - 1:
                    done = True

                agent.push_memory(st, a, r, (not done), st1)

                if done:
                    break

                st = st1

            # TODO: decrease epsilon for multiple trajectories evaluations ???
            agent.epsilon *= agent.epsilon_decay

        tot_reward = sim.normalize_reward(tot_reward, params['path_length'])
        agent.write_reward(tot_reward, final_r)

        # Update the networks
        l_i = agent.update_Q(theta_i)
        l_tot += l_i

        if epoch % params['batch_tasks'] == params['batch_tasks'] - 1:
            # backprop from the mean of the TD losses in the batch
            agent.optimizer.zero_grad()
            l_tot.backward()
            agent.optimizer.step()
            l_tot = 0

            ''' TEST NEW POLICY ON CURRENT MAZE '''

            # sim.reset()
            # s0 = sim.get_state()
            # a0 = agent.get_action(s0)
            # r0, done = sim.step(a0)
            # s1 = sim.get_state()
            theta_i = get_adapted_theta(agent, sim) # agent.adapt(s0, a0, r0, (not done), s1)

            tot_reward = 0

            sim.reset()
            st = sim.get_state()

            for t in range(path_len):
                a = agent.get_action(st, theta=theta_i, test=True)
                r, done = sim.step(a)

                st1 = sim.get_state()

                tot_reward += r

                if done:
                    break

                st = st1

            tot_reward = sim.normalize_reward(tot_reward, params['path_length'])
            agent.writer.add_scalar("Test reward", tot_reward, int(epoch/params['batch_tasks']))

        ''' TEST ON OLD MAZES '''

        # Once trained in a new maze, test the performances in the previous mazes.
        if epoch > 0:
            tot_reward = 0
            first_maze = max(0, epoch-20)
            diff_mazes = epoch - first_maze
            for i, temp_maze in enumerate(mazes[first_maze:epoch]):

                x = first_maze + i

                sim = Simulator(starts[x], goals[x], temp_maze, params)
                T = int(paths_length[x] * params['horizon_multiplier'])
                # sim.reset()
                # s0 = sim.get_state()
                # a0 = agent.get_action(s0)
                # r0, done = sim.step(a0)
                # s1 = sim.get_state()
                theta_i = get_adapted_theta(agent, sim) #agent.adapt(s0, a0, r0, (not done), s1)

                sim.reset()
                st = sim.get_state()
                tmp_reward = 0

                for t in range(paths_length[x]):

                    a = agent.get_action(st, theta=theta_i, test=True)
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

        rnd_start, rnd_goal, rnd_maze, rnd_paths_length, _ = maze_gen.get_maze(central=False)
        rnd_sim = Simulator(rnd_start, rnd_goal, rnd_maze, params)
        T = rnd_paths_length * params['horizon_multiplier']

        theta_i = get_adapted_theta(agent, rnd_sim)

        tot_reward = 0

        rnd_sim.reset()
        st = rnd_sim.get_state()

        for t in range(rnd_paths_length):
            a = agent.get_action(st, theta=theta_i, test=True)
            r, done = rnd_sim.step(a)

            st1 = rnd_sim.get_state()

            tot_reward += r

            if done:
                break

            st = st1

        tot_reward = rnd_sim.normalize_reward(tot_reward, params['path_length'])
        agent.writer.add_scalar("Test reward on random starting point", tot_reward, int(epoch))


    print()
