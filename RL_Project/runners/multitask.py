import numpy as np
from tqdm import tqdm

from RL_Project.agents.dqn_agent import DQN
from RL_Project.agents.pg_agents import REINFORCE, PPO
from RL_Project.environment import Simulator
# from agents.dqn_agent import DQN
# from agents.pg_agents import REINFORCE, PPO
# from environment import Simulator


def multitask(params, logdir, device):

    mazes = None
    paths_length = []

    if params['agent'] == 'dqn':
        agent = DQN(params, logdir, device).to(device)
    elif params['agent'] == 'reinforce':
        agent = REINFORCE(params, logdir, device).to(device)
    elif params['agent'] == 'ppo':
        agent = PPO(params, logdir, device).to(device)

    # train over multiple MDPs batches if mbs is grater than 1
    for epoch in range(params['epochs']):

        sims = []
        Ts = []

        # frq = np.zeros((500, 12, 12))
        # frq_idx = 0

        for i in range(params['mbs']):

            sim = Simulator(params)
            sims.append(sim)
            grid = np.expand_dims(sim.grid.copy(), 0)
            paths_length.append(sim.T)
            Ts.append(sim.T)
            if mazes is None:
                mazes = grid
            else:
                mazes = np.concatenate((mazes, grid), 0)

        ''' TEST AGENT ON NEW MAZES '''

        avg_cumulative_reward = 0
        avg_final_reward = 0
        for i in range(params['mbs']):
            sims[i].reset()
            st = [sims[i].get_state() for i in range(params['mbs'])]
            mdp_reward = 0
            for t in range(Ts[i]):
                a, _ = agent.get_action(st[i], test=True)
                r, done = sims[i].step(a)
                st1 = sims[i].get_state()

                mdp_reward += r
                if done:
                    break
                st[i] = st1
            avg_cumulative_reward += mdp_reward
            avg_final_reward += r
        agent.writer.add_scalar("Test cumulative reward new mazes", avg_cumulative_reward / params['mbs'], epoch)
        agent.writer.add_scalar("Test final reward new mazes", avg_final_reward / params['mbs'], epoch)

        agent.epsilon = agent.epsilon0

        # start training of the maze
        for e in tqdm(range(params['episodes'])):

            for i in range(params['mbs']):

                sims[i].reset()
                st = sims[i].get_state()
                T = Ts[i] * params['horizon_multiplier']

                for t in range(T):
                    # frqs[i, sim[i].actual_pos_x, sim[i].actual_pos_y] += 1
                    a, logprob = agent.get_action(st)
                    r, done = sims[i].step(a)

                    st1 = sims[i].get_state()

                    # frq[frq_idx] += st1[0, 1]

                    if t == T - 1:
                        done = True

                    agent.push_data(st, a, logprob, r, done, st1)

                    if done:
                        break

                    st = st1

            # Update the networks
            agent.update(e)

            # perform a test of the policy where there is no exploration
            if e % params['episodes_test'] == params['episodes_test']-1:

                # plt.figure()
                # plt.imshow(frq - st1[0, 0])
                # plt.show()

                # frq_idx += 1

                avg_cumulative_reward = 0
                avg_final_reward = 0

                for i in range(params['mbs']):

                    sims[i].reset()
                    st = sims[i].get_state()
                    mdp_reward = 0
                    T = Ts[i]

                    for t in range(T):
                        a, _ = agent.get_action(st, test=True)
                        r, done = sims[i].step(a)

                        st1 = sims[i].get_state()

                        mdp_reward += r

                        if done:
                            break

                        st = st1

                    avg_cumulative_reward += mdp_reward
                    avg_final_reward += r

                # print(avg_final_reward / params['mbs'])

                agent.write_reward(avg_cumulative_reward / params['mbs'], avg_final_reward / params['mbs'])






        for i in range(params['mbs']):
            sims[i].reset()

            frq = np.zeros((12, 12))


            st = [sims[i].get_state() for i in range(params['mbs'])]
            mdp_reward = 0
            for t in range(Ts[i]):
                a, _ = agent.get_action(st[i], test=True)
                r, done = sims[i].step(a)
                st1 = sims[i].get_state()

                frq += st1[0, 1]

                mdp_reward += r
                if done:
                    break
                st[i] = st1
            avg_cumulative_reward += mdp_reward
            avg_final_reward += r

        print()





        ''' TEST ON OLD MAZES '''

        # Once trained in a new maze, test the performances in the previous mazes.
        if epoch > 0:
            avg_cumulative_reward = 0
            avg_final_reward = 0

            first_maze = max(0, epoch - 20)
            diff_mazes = epoch - first_maze

            for i, temp_maze in enumerate(mazes[first_maze:epoch]):
                x = first_maze + i

                sim = Simulator(params, grid=mazes[x], T=paths_length[x])
                T = paths_length[x]

                st = sim.get_state()
                tmp_reward = 0

                for t in range(T):

                    a, _ = agent.get_action(st, test=True)
                    r, done = sim.step(a)

                    st1 = sim.get_state()
                    tmp_reward += r

                    if done:
                        break
                    st = st1

                avg_cumulative_reward += tmp_reward
                avg_final_reward += r

            agent.writer.add_scalar("Test cumulative reward past mazes", avg_cumulative_reward / diff_mazes, int(epoch))
            agent.writer.add_scalar("Test final reward past mazes", avg_final_reward / diff_mazes, int(epoch))

    print()


