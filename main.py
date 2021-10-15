from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from scipy import interpolate
import argparse
from runners.dqn import dqn
from runners.meta_dqn import meta_dqn
from runners.meta_pg import meta_pg
from runners.pg_meta_inverse import inverse_meta_pg
from runners.pg import pg


parser = argparse.ArgumentParser()
parser.add_argument('--agent', default='ppo', type=str, help='Type of agent, either dqn or ppo or reinforce')
parser.add_argument('--maml', default=True, type=bool, help='Maml agent or not')


parser.add_argument('--seed', default=1234, type=int, help='Seed')


parser.add_argument('--mbs', default=1, type=int, help='Number of mazes per batch')
parser.add_argument('--episodes_test', default=1000, type=int, help='Number of episodes per epoch during test')
parser.add_argument('--episodes_maml', default=100, type=int, help='Number of episodes per epoch during training in MAML') # todo: useless


parser.add_argument('--batch_tasks', default=50, type=int, help='Number of mazes per batch during maml training')


parser.add_argument('--horizon_multiplier_adaptation', default=9, type=int, help='Multiplier of shortest path size to define max steps per episode for the adaptation step')


parser.add_argument('--modalities_goal_dist', default=0, type=int, help='0: uniform, else number of possible different goal positions')


parser.add_argument('--epochs', default=100000, type=int, help='Number of different mazes to train on')
parser.add_argument('--adaptive_lr', default=False, type=bool, help='Per parameter adaptive learning rate')
parser.add_argument('--horizon_multiplier', default=1, type=int, help='Multiplier of shortest path size to define max steps per episode')  # todo: back to 3
parser.add_argument('--adaptation_trajectories', default=24, type=int, help='Number of trajectories used for the adaptation step (MAML), 0 means only one step') # todo: back to 25 without optim
parser.add_argument('--eps_adapt_decay', default=0.8, type=float, help='epsilon adaptation decay')      #todo: 0.6 for 10 trajs
parser.add_argument('--adaptation_best_trajectories', default=1.0, type=float, help='Ratio of trajectories with highest reword to be used in adaptation')
parser.add_argument('--adaptation_optimal_traj', default=False, type=bool, help='Sample from optimal paths for adaptation')
parser.add_argument('--episodes', default=10, type=int, help='Number of episodes per epoch during training')  # todo: back to 100? eps decay back to 0.98?
parser.add_argument('--episode_per_update', default=5, type=int, help='Number of episodes before updating PPO: K') # 10
parser.add_argument('--regression_steps', default=10, type=int, help='number of updates in the outer loop of inverse maml')
parser.add_argument('--show_goal', default=0, type=int, help='0: state=obstacles+position, 1: state=obstacles+position+goal')
parser.add_argument('--filter_type', default=0, type=int, help='0: no filter, 1: max_reward, 2: state frequencies, 3: state action frequencies, 4: forward curiosity, 5: max inner loss')
parser.add_argument('--activation', default='tanh', type=str, help='either tanh or relu') #TODO: tanh
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='either 1e-5 or 1e-8')
parser.add_argument('--norm_A', default=1, type=int, help='normalize advantage in the outer loop for meta PPO')
parser.add_argument('--c1', default=0.5, type=float, help='scaling constant for value loss in PPO')
parser.add_argument('--c2', default=0.0, type=float, help='scaling constant for entropy bonus in PPO')
parser.add_argument('--gradient_clipping', default=1, type=int, help='clip gradients in PPO')


parser.add_argument('--cl2', default=1.0, type=float, help='scaling constant additive loss')
parser.add_argument('--reg_l2', default=0, type=int, help='regularization additive loss')


parser.add_argument('--eps_clip', default=0.2, type=float, help='ppo epsilon clipping term')


parser.add_argument('--inner_lr', default=0.1, type=float, help='inner loss learning rate')
parser.add_argument('--lr', default=0.001, type=float, help='outer loss learning rate')
parser.add_argument('--curiosity_lr', default=0.0001, type=float, help='curiosity module learning rate')
parser.add_argument('--kl_nu', default=0.0005, type=float, help='kl scale factor')


parser.add_argument('--gradient_alignment', default=1, type=int, help='add cosine similarity btw gradients in the outer loss')


parser.add_argument('--grid_size', default=15, type=int, help='Size of the grid of the environment')
parser.add_argument('--path_length', default=6, type=int, help='Minimum distance to goal')
parser.add_argument('--pos_val', default=1, type=int, help='Value in the maze which represents the position of the agent')
parser.add_argument('--goal_val', default=1, type=int, help='Value in the maze which represents the goal of the agent')
parser.add_argument('--obs_val', default=1, type=int, help='Value in the maze which represents the obstacles')


parser.add_argument('--p_obstacles', default=0., type=float, help='probability each cell becomes an obstacle')
parser.add_argument('--sparse_reward', default=1, type=int, help='0: dense reward, 1: sparse reward')


parser.add_argument('--add_loss_exploration', default=0, type=int, help='0: no l2, 1: l2 curiosity')  # , 2: l2 trajectory prediction
parser.add_argument('--inner_loss_type', default=1, type=int, help='0: PPO, 1: reward prediction')
parser.add_argument('--decoupled_optimization', default=0, type=int, help='0: same optimization for exploration and exploitation, 1: different optimization')
parser.add_argument('--decoupled_explorer', default=1, type=int, help='0: same model for exploration and exploitation, 1: different model')
parser.add_argument('--explorer_loss', default=1, type=int, help='0: reinforce with dyn. pred. cumul. error, 1: policy entropy, 2: PPO with prediction error and entropy')

# This is to have just one reward predictor and to connect beta to theta
parser.add_argument('--decoupled_predictors', default=0, type=int, help='0: same network for inner and outer prediction, 1: different networks')
parser.add_argument('--beta_model', default=1, type=int, help='0: input = z_0 latent space and action, 1: different networks')


parser.add_argument('--exploiter_iteration', default=3, type=int, help='Number of exploiter optimization') # TODO: 10

if __name__ == '__main__':
    args = parser.parse_args()
    params = args.__dict__

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # logdir = './debug_logs/path_length='+str(params['path_length'])+'_N_trj_adapt='+str(params['adaptation_trajectories'])+'_eps_adapt='+str(params['eps_adapt'])
    # logdir = './logs_grad_alg/2_OK_100'
    # logdir = './logs_filters_nme/grid_size='+str(params['grid_size'])
    # logdir += '_path_length=' + str(params['path_length'])
    # logdir += '_batch_tasks=' + str(params['batch_tasks'])
    # logdir += '_episodes_maml=' + str(params['episodes_maml'])
    # logdir += '_show_goal=' + str(params['show_goal'])
    # logdir += '_adaptation_trajectories=' + str(params['adaptation_trajectories'])
    # logdir += '_adaptation_best_trajectories=' + str(params['adaptation_best_trajectories'])
    # logdir += '_eps_adapt=' + str(params['eps_adapt'])
    # logdir += '_modalities_goal_dist=' + str(params['modalities_goal_dist'])
    # logdir += '_filter_type=' + str(params['filter_type'])

    if params['inner_loss_type'] == 0 and params['decoupled_explorer'] == 1:
        exit()
    if params['decoupled_optimization'] == 1 and (params['add_loss_exploration'] == 0 or params['decoupled_explorer'] == 0):
        exit()
    if params['decoupled_explorer'] == 0 and params['exploiter_iteration'] > 1:
        exit()
    if params['inner_loss_type'] == 0 and params['decoupled_predictors'] == 0:
        exit()

    # writer = SummaryWriter(log_dir="./logs_inverse_maml/dense_r")
    # inverse_meta_pg(params, "./delete", device, writer)
    # exit()

    # logdir = './logs_promp9/l2='+str(params['add_loss_exploration'])+'_l_inner='+str(params['inner_loss_type'])+'_decouple_e='+str(params['decoupled_explorer'])+'_decouple_opt='+str(params['decoupled_optimization'])+'_exploiter_it='+str(params['exploiter_iteration'])
    # logdir = './logs_promp3/l2='+str(params['add_loss_exploration'])+'_inner_type='+str(params['inner_loss_type'])+'_decouple='+str(params['decoupled_explorer'])
    # logdir = './logs_promp10/sparse=' + str(params['sparse_reward']) + '_loss=' + str(params['explorer_loss']) + '_decouple_opt=' + str(params['decoupled_optimization']) + '_exploiter_it=' + str(params['exploiter_iteration'])
    logdir = './logs_promp11/exploiter_it=' + str(params['exploiter_iteration']) + '_seed=' + str(params['seed'])

    # logdir = "./debug/3"
    writer = SummaryWriter(log_dir=logdir)

    n_experiment = ""

    meta_pg(params, writer, n_experiment, device)
    exit()

    # params['epochs'] = 200
    #
    # for l2 in [0, 1]:
    #     for l_inner in [0, 1]:
    #         for decouple in [0, 1]:
    #             params['add_loss_exploration'] = l2
    #             params['inner_loss_type'] = l_inner
    #             params['decoupled_explorer'] = decouple
    #
    #             if params['inner_loss_type'] == 0 and params['decoupled_explorer'] == 1:
    #                 continue
    #
    #             logdir = './logs_promp/l2='+str(l2)+'_l_inner='+str(l_inner)+'_decouple='+str(decouple)
    #             writer = SummaryWriter(log_dir=logdir)
    #
    #             meta_pg(params, writer, device)
    #
    # exit()


    # if params['maml']:
    #     params['episodes'] = params['episodes_maml']
    #
    # if params['agent'] == 'dqn':
    #     if params['maml']:
    #         meta_dqn(params, logdir, device)
    #     else:
    #         dqn(params, logdir, device)
    #
    if params['agent'] == 'reinforce' or params['agent'] == 'ppo':
        if params['maml']:
            meta_pg(params, writer, device)
        else:
            pg(params, logdir, device)

    # inverse_meta_pg(params, logdir, device, writer)

    # if params['agent'] == 'ppo':
    #     if params['maml']:
    #         meta_PPO(params, logdir, device)
    #     else:
    #         pg(params, logdir, device)

    print()


# def plot_data(data_vector):
#     tmp_data = data_vector
#     avg = 5
#     data_mean = np.zeros(int(len(tmp_data)/avg))
#     data_max = np.zeros(int(len(tmp_data)/avg))
#     data_min = np.zeros(int(len(tmp_data)/avg))
#     for i in range(int(len(tmp_data)/avg)):
#         data_mean[i] = np.mean(np.array(tmp_data[i:i + avg]))
#         data_max[i] = np.max(np.array(tmp_data[i:i + avg]))
#         data_min[i] = np.min(np.array(tmp_data[i:i + avg]))
#     x = list(range(len(data_mean)))
#     plt.plot(x, data_mean)
#     plt.fill_between(x, data_min, data_max, alpha=0.3)
# plt.figure()
# plot_data(l0_inner0_dec0_sparse)
# plot_data(l0_inner1_dec0_sparse)
# plot_data(l2_inner0_dec0_sparse)
# plot_data(l2_inner1_dec0_sparse)
# plt.legend(['pg', 'r_pred', 'l2_pg', 'l2_r_pred'])


