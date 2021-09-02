import numpy as np
from env import Simulator
from agents.dqn_agent import Net
from maze_gen import Maze_Gen
from tqdm import tqdm


def dqn(params, logdir, device):

    mazes = None
    paths_length = []
    maze_gen = Maze_Gen(params)

    agent = Net(params, logdir, device).to(device)

    # train over multiple MDPs batches if mbs is grater than 1
    for epoch in range(params['epochs']):

        for i in range(params['mbs']):
            start, goal, maze, path_len, _ = maze_gen.get_maze()
            maze = np.expand_dims(maze, 0)
            paths_length.append(path_len)
            if mazes is None:
                mazes = maze
                starts = [start]
                goals = [goal]
            else:
                mazes = np.concatenate((mazes, maze), 0)
                starts.append(start)
                goals.append(goal)

        batch_mazes = mazes[epoch * params['mbs']:(epoch + 1) * params['mbs']]
        batch_goals = goals[epoch * params['mbs']:(epoch + 1) * params['mbs']]
        batch_starts = starts[epoch * params['mbs']:(epoch + 1) * params['mbs']]
        batch_path_lengths = paths_length[epoch * params['mbs']:(epoch + 1) * params['mbs']]
        sim = [Simulator(batch_starts[q], batch_goals[q], batch_mazes[q], params) for q in range(params['mbs'])]
        T = np.max([(batch_path_lengths[q] * params['horizon_multiplier']) for q in range(params['mbs'])])
        frqs = np.zeros((params['mbs'], params['grid_size']+2, params['grid_size']+2))

        ''' TEST AGENT ON NEW MAZES '''

        tot_reward = 0
        for i in range(params['mbs']):
            sim[i].reset()
            st = [sim[i].get_state() for i in range(params['mbs'])]
            mdp_reward = 0
            for t in range(batch_path_lengths[i]):
                a = agent.get_action(st[i], test=True)
                r, done = sim[i].step(a)
                st1 = sim[i].get_state()
                mdp_reward += r
                if done:
                    break
                st[i] = st1
            mdp_reward = sim[i].normalize_reward(mdp_reward, params['path_length'])
            tot_reward += mdp_reward
        agent.writer.add_scalar("Test reward before training", tot_reward / params['mbs'], epoch)

        ''' TRAIN ON NEW MAZES '''

        agent.epsilon = agent.epsilon0

        # start training of the maze
        for e in tqdm(range(params['episodes'])):

            tot_reward = 0
            final_r = 0

            for i in range(params['mbs']):

                sim[i].reset()
                st = [sim[k].get_state() for k in range(params['mbs'])]
                mdp_reward = 0

                for t in range(T):
                    frqs[i, sim[i].actual_pos_x, sim[i].actual_pos_y] += 1
                    a = agent.get_action(st[i])
                    r, done = sim[i].step(a)

                    st1 = sim[i].get_state()

                    mdp_reward += r
                    final_r = r

                    if t == T - 1:
                        done = True

                    agent.push_memory(st[i], a, r, (not done), st1)

                    if done:
                        break

                    st[i] = st1

                mdp_reward = sim[i].normalize_reward(mdp_reward, params['path_length'])
                tot_reward += mdp_reward

            # Update the networks
            agent.update_Q()
            agent.update_target(e)
            agent.write_reward(tot_reward/params['mbs'], final_r/params['mbs'])

            # perform a test of the policy where there is no exploration
            if e % params['episodes_test'] == params['episodes_test']-1:

                tot_reward = 0

                for i in range(params['mbs']):

                    sim[i].reset()
                    st = [sim[i].get_state() for i in range(params['mbs'])]
                    mdp_reward = 0

                    for t in range(batch_path_lengths[i]):
                        a = agent.get_action(st[i], test=True)
                        r, done = sim[i].step(a)

                        st1 = sim[i].get_state()

                        mdp_reward += r

                        if done:
                            break

                        st[i] = st1

                    mdp_reward = sim[i].normalize_reward(mdp_reward, params['path_length'])
                    tot_reward += mdp_reward

                agent.writer.add_scalar("Test reward during training", tot_reward/params['mbs'], int(epoch * (params['episodes']/params['episodes_test']) + e / params['episodes_test']))

        ''' TEST ON OLD MAZES '''

        # Once trained in a new maze, test the performances in the previous mazes.
        if epoch > 0:
            tot_reward = 0
            for x, temp_maze in enumerate(mazes[:epoch*params['mbs']]):

                sim = Simulator(starts[x], goals[x], temp_maze, params)
                T = int(paths_length[x] * params['horizon_multiplier'])

                st = sim.get_state()
                tmp_reward = 0

                for t in range(paths_length[x]):

                    a = agent.get_action(st, test=True)
                    r, done = sim.step(a)

                    st1 = sim.get_state()
                    tmp_reward += r

                    if done:
                        break
                    st = st1

                tmp_reward = sim.normalize_reward(tmp_reward, params['path_length'])
                tot_reward += tmp_reward
                print('maze: ' + str(x) + ' reward: ' + str(tmp_reward), end=' ')
            print()

            agent.writer.add_scalar("Previous mazes average reward", tot_reward / (epoch*params['mbs']), int(epoch))

        ''' TEST IN RANDOM STARTING POINT AND MAZE'''

        rnd_start, rnd_goal, rnd_maze, rnd_paths_length, _ = maze_gen.get_maze(central=False)
        rnd_sim = Simulator(rnd_start, rnd_goal, rnd_maze, params)
        T = rnd_paths_length * params['horizon_multiplier']

        tot_reward = 0

        rnd_sim.reset()
        st = rnd_sim.get_state()

        for t in range(rnd_paths_length):
            a = agent.get_action(st, test=True)
            r, done = rnd_sim.step(a)

            st1 = rnd_sim.get_state()

            tot_reward += r

            if done:
                break

            st = st1

        agent.writer.add_scalar("Test reward on random starting point", rnd_sim.normalize_reward(tot_reward, params['path_length']), int(epoch))

    print()


