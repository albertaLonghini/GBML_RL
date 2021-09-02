import numpy as np
import torch
import time


def random_filter(N_trj, ratio_best_trj):

    indeces = list(np.array(range(int(N_trj))))

    return indeces


def max_reward(cumulative_rwds):

    indeces = np.argsort(cumulative_rwds)

    return indeces


def state_frequencies(trajectories, use_actions):

    indeces = []

    if not use_actions:
        trj_pos = [np.concatenate([x[6][:, 1] for x in tau], 0) for tau in trajectories]
    else:
        # TODO: works only for 4 possible actions
        trj_pos = [np.concatenate([np.concatenate([np.expand_dims(x[0][:, 1], 1) * (x[1] == 0) * 1,
                                                   np.expand_dims(x[0][:, 1], 1) * (x[1] == 1) * 1,
                                                   np.expand_dims(x[0][:, 1], 1) * (x[1] == 2) * 1,
                                                   np.expand_dims(x[0][:, 1], 1) * (x[1] == 3) * 1], 1) for x in tau], 0) for tau in trajectories]

    tot_frq = np.sum(np.concatenate(trj_pos, 0), 0)
    trj_frqs = [tot_frq[np.nonzero(pos)[1:]] for pos in trj_pos]
    score_frqs = [np.sum(x ** 2) for x in trj_frqs]
    first_idx = np.argsort(score_frqs)[0]
    indeces.append(first_idx)

    trj_frq_img = [np.sum(pos, 0) for pos in trj_pos]

    relative_frq = trj_frq_img[first_idx]

    remaining_taus = list(range(len(trajectories)))
    del remaining_taus[first_idx]

    while len(remaining_taus) > 0:
        relative_trj_frq = [relative_frq[np.nonzero(pos)[1:]] for pos in [trj_pos[i] for i in remaining_taus]]
        score_frqs = [np.sum(x ** 2) for x in relative_trj_frq]
        relative_idx = np.argsort(score_frqs)[0]
        next_idx = remaining_taus[relative_idx]
        relative_frq += trj_frq_img[next_idx]
        indeces.append(next_idx)
        del remaining_taus[relative_idx]

    indeces.reverse()

    return indeces


def forward_prediction(agent, trajectories):

    states = torch.cat([torch.cat([agent.to_tensor(x[0]) for x in tau], 0) for tau in trajectories], 0)
    actions = torch.cat([torch.cat([agent.to_tensor(x[1]).view(1, 1) for x in tau], 0) for tau in trajectories], 0)
    next_states = torch.cat([torch.cat([agent.to_tensor(x[6]) for x in tau], 0) for tau in trajectories], 0)

    z = agent.forward_model.phi(states)
    z1 = agent.forward_model.phi(next_states)
    z1_tilde = agent.forward_model(z, actions)

    forward_err = torch.sum((z1 - z1_tilde) ** 2, -1)

    len_trjs = [len(trj) for trj in trajectories]
    cum_len_trj = np.cumsum(len_trjs)
    cum_len_trj = np.concatenate([np.array([0]), cum_len_trj])
    forward_err_trjs = [torch.sum(forward_err[cum_len_trj[i]:cum_len_trj[i + 1]]).detach().cpu().item() for i in range(len(cum_len_trj) - 1)]

    indeces = np.argsort(np.array(forward_err_trjs))

    agent.optimizer_forward.zero_grad()
    forward_err.mean().backward()
    agent.optimizer_forward.step()

    return indeces


def max_inner_loss(agent, trajectories):

    with torch.no_grad():

        rewards = [[x[4]for x in tau] for tau in trajectories]
        logprobs = [torch.cat([x[2] for x in tau], 0) for tau in trajectories]

        cumulative_rewards = []
        for trj_r in rewards:
            tmp_r = []
            tmp_r.append(trj_r[-1])
            for i in range(len(trj_r) - 1, 0, -1):
                tmp_r.append(trj_r[i] + agent.gamma * tmp_r[-1])
            tmp_r.reverse()
            cumulative_rewards.append(agent.to_tensor(tmp_r))

        score = np.array([(-R * lp).mean().detach().cpu().item() for R, lp in zip(cumulative_rewards, logprobs)])
        indeces = list(np.argsort(score))

        indeces.reverse()

        return indeces



















