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
# parser.add_argument('--agent', default='ppo', type=str, help='Type of agent, either dqn or ppo or reinforce')
# parser.add_argument('--maml', default=True, type=bool, help='Maml agent or not')





parser.add_argument('--seed', default=1234, type=int, help='Seed')

# parser.add_argument('--mbs', default=1, type=int, help='Number of mazes per batch')
# parser.add_argument('--episodes_test', default=1000, type=int, help='Number of episodes per epoch during test')

# parser.add_argument('--batch_tasks', default=10, type=int, help='Number of mazes per batch during maml training')
# parser.add_argument('--episodes_maml', default=100, type=int, help='Number of episodes per epoch during training in MAML') # todo: useless




# parser.add_argument('--horizon_multiplier_adaptation', default=1, type=int, help='Multiplier of shortest path size to define max steps per episode for the adaptation step')







parser.add_argument('--modalities_goal_dist', default=4, type=int, help='0: unifrom, else number of possible different goal positions')









parser.add_argument('--epochs', default=100000, type=int, help='Number of different mazes to train on')
parser.add_argument('--adaptive_lr', default=False, type=bool, help='Per parameter adaptive learning rate')
parser.add_argument('--horizon_multiplier', default=1, type=int, help='Multiplier of shortest path size to define max steps per episode')  # todo: back to 3
parser.add_argument('--adaptation_trajectories', default=10, type=int, help='Number of trajectories used for the adaptation step (MAML), 0 means only one step')
parser.add_argument('--eps_adapt', default=0.2, type=float, help='epsilon adaptation')
parser.add_argument('--adaptation_best_trajectories', default=1.0, type=float, help='Ratio of trajectories with highest reword to be used in adaptation')
parser.add_argument('--adaptation_optimal_traj', default=False, type=bool, help='Sample from optimal paths for adaptation')
parser.add_argument('--episodes', default=10000, type=int, help='Number of episodes per epoch during training')
parser.add_argument('--episode_per_update', default=50, type=int, help='Number of episodes before updating PPO: K') # 10
parser.add_argument('--regression_steps', default=10, type=int, help='number of updates in the outer loop of inverse maml')
parser.add_argument('--show_goal', default=0, type=int, help='0: state=obstacles+position, 1: state=obstacles+position+goal')
parser.add_argument('--filter_type', default=0, type=int, help='0: no filter, 1: max_reward, 2: state frequencies, 3: state action frequencies, 4: forward curiosity, 5: max inner loss')
parser.add_argument('--model_type', default=0, type=int, help='Model used to train the dqn agent. Up to now it changes the depth of the network[its a flag, either 0 or 1')
parser.add_argument('--activation', default='leaky_relu', type=str, help='either tanh or relu')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='either 1e-5 or 1e-8')
parser.add_argument('--norm_A', default=1, type=int, help='normalize advantage in the outer loop for meta PPO')
parser.add_argument('--c1', default=0.5, type=float, help='scaling constant for value loss in PPO')
parser.add_argument('--c2', default=0.0, type=float, help='scaling constant for entropy bonus in PPO')
parser.add_argument('--gradient_clipping', default=1, type=int, help='clip gradients in PPO')

parser.add_argument('--grid_size', default=15, type=int, help='Size of the grid of the environment')
parser.add_argument('--path_length', default=7, type=int, help='Minimum distance to goal')
parser.add_argument('--pos_val', default=1, type=int, help='Value in the maze which represents the position of the agent')
parser.add_argument('--goal_val', default=1, type=int, help='Value in the maze which represents the goal of the agent')
parser.add_argument('--obs_val', default=1, type=int, help='Value in the maze which represents the obstacles')

if __name__ == '__main__':
    args = parser.parse_args()
    params = args.__dict__

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # logdir = './debug_logs/path_length='+str(params['path_length'])+'_N_trj_adapt='+str(params['adaptation_trajectories'])+'_eps_adapt='+str(params['eps_adapt'])
    logdir = './logs/inverse_maml_grid_size=15_path_len=7_leaky_relu'
    writer = SummaryWriter(log_dir=logdir)
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

    # if params['maml']:
    #     params['episodes'] = params['episodes_maml']
    #
    # if params['agent'] == 'dqn':
    #     if params['maml']:
    #         meta_dqn(params, logdir, device)
    #     else:
    #         dqn(params, logdir, device)
    #
    # if params['agent'] == 'reinforce' or params['agent'] == 'ppo':
    #     if params['maml']:
    #         meta_pg(params, logdir, device)
    #     else:
    #         pg(params, logdir, device)

    inverse_meta_pg(params, logdir, device, writer)

    # if params['agent'] == 'ppo':
    #     if params['maml']:
    #         meta_PPO(params, logdir, device)
    #     else:
    #         pg(params, logdir, device)

    print()



