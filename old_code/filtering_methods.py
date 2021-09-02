import numpy as np


def max_reward(agent, temp_data, mode, idx, N_trj, ratio_best_trj, cumulative_rwds, temp_num_optim_trj):

    if mode == agent.ADAPTATION:
        indeces = np.argsort(cumulative_rwds)
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
            agent.push_batchdata(batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4], batch_data[5], mode, idx)

    return tot_rwd_trj / N_best_trj, num_optim_trj
























