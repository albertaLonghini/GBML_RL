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
from runners.pg import pg
from runners.meta_PPO import meta_PPO


parser = argparse.ArgumentParser()
parser.add_argument('--agent', default='ppo', type=str, help='Type of agent, either dqn or ppo or reinforce')
parser.add_argument('--maml', default=True, type=bool, help='Maml agent or not')

parser.add_argument('--grid_size', default=15, type=int, help='Size of the grid of the environment')
parser.add_argument('--path_length', default=2, type=int, help='Minimum distance to goal')
parser.add_argument('--epochs', default=100000, type=int, help='Number of different mazes to train on')
parser.add_argument('--horizon_multiplier', default=1, type=int, help='Multiplier of shortest path size to define max steps per episode')  # todo: back to 3
parser.add_argument('--episodes', default=10000, type=int, help='Number of episodes per epoch during training')
parser.add_argument('--pos_val', default=1, type=int, help='Value in the maze which represents the position of the agent')
parser.add_argument('--goal_val', default=1, type=int, help='Value in the maze which represents the goal of the agent')
parser.add_argument('--obs_val', default=1, type=int, help='Value in the maze which represents the obstacles')
parser.add_argument('--seed', default=1234, type=int, help='Seed')

parser.add_argument('--mbs', default=1, type=int, help='Number of mazes per batch')
parser.add_argument('--episodes_test', default=1000, type=int, help='Number of episodes per epoch during test')

parser.add_argument('--batch_tasks', default=100, type=int, help='Number of mazes per batch during maml training')
parser.add_argument('--episodes_maml', default=100, type=int, help='Number of episodes per epoch during training in MAML')

parser.add_argument('--episode_per_update', default=50, type=int, help='Number of episodes before updating PPO') # 10

parser.add_argument('--adaptation_trajectories', default=100, type=int, help='Number of trajectories used for the adaptation step (MAML), 0 means only one step')
parser.add_argument('--horizon_multiplier_adaptation', default=1, type=int, help='Multiplier of shortest path size to define max steps per episode for the adaptation step')
parser.add_argument('--adaptation_best_trajectories', default=1.0, type=float, help='Ratio of trajectories with highest reword to be used in adaptation')
parser.add_argument('--adaptation_optimal_traj', default=False, type=bool, help='Sample from optimal paths for adaptation')

parser.add_argument('--adaptive_lr', default=False, type=bool, help='Per parameter adaptive learning rate')

parser.add_argument('--model_type', default=0, type=int, help='Model used to train the dqn agent. Up to now it changes the depth of the network[its a flag, either 0 or 1')
parser.add_argument('--activation', default='tanh', type=str, help='either tanh or relu')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='either 1e-5 or 1e-8')
parser.add_argument('--norm_A', default=1, type=int, help='normalize advantage in the outer loop for meta PPO')
parser.add_argument('--c1', default=0.5, type=float, help='scaling constant for value loss in PPO')
parser.add_argument('--c2', default=0.0, type=float, help='scaling constant for entropy bonus in PPO')
parser.add_argument('--gradient_clipping', default=1, type=int, help='clip gradients in PPO')

parser.add_argument('--eps_adapt', default=1., type=float, help='epsilon adaptation')

if __name__ == '__main__':
    args = parser.parse_args()
    params = args.__dict__

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # logdir = './meta_ppo/openAI_tricks_part_1_grid_size=15_c1=0.5'#episodes_per_update='+str(params['episode_per_update'])+"_adaptation_trajectories="+str(params['adaptation_trajectories'])+"_c1=0.1"

    # logdir = './logs_ppo/batch_tasks='+str(params['batch_tasks'])+'_model_type='+str(params['model_type'])+'_activation_f='+params['activation']+'_adam_eps='+str(params['adam_epsilon'])+'_norm_A='+str(params['norm_A'])+'_clip_grads='+str(params['gradient_clipping'])

    logdir = './logs_ppo_2/path_length='+str(params['path_length'])+'_N_trj_adapt='+str(params['adaptation_trajectories'])+'_eps_adapt='+str(params['eps_adapt'])

    if params['maml']:
        params['episodes'] = params['episodes_maml']

    if params['agent'] == 'dqn':
        if params['maml']:
            meta_dqn(params, logdir, device)
        else:
            dqn(params, logdir, device)

    if params['agent'] == 'reinforce':
        if params['maml']:
            meta_pg(params, logdir, device)
        else:
            pg(params, logdir, device)

    if params['agent'] == 'ppo':
        if params['maml']:
            meta_PPO(params, logdir, device)
        else:
            pg(params, logdir, device)

    print()



