import torch
import numpy as np
import argparse
from RL_Project.runners.multitask import multitask
from RL_Project.runners.meta_learning import meta_learning
# from runners.multitask import multitask
# from runners.meta_learning import meta_learning


parser = argparse.ArgumentParser()
parser.add_argument('--agent', default='ppo', type=str, help='Type of agent, either dqn or ppo or reinforce')
parser.add_argument('--maml', default=True, type=bool, help='Maml agent or not')

parser.add_argument('--epochs', default=100000, type=int, help='Number of different mazes to train on') #100000
parser.add_argument('--episodes', default=10000, type=int, help='Number of episodes per epoch during training')
parser.add_argument('--seed', default=1234, type=int, help='Seed')
parser.add_argument('--horizon_multiplier', default=2, type=int, help='Multiplier of shortest path size to define max steps per episode')  # todo: back to 3

parser.add_argument('--mbs', default=1, type=int, help='Number of mazes per batch')
parser.add_argument('--episodes_test', default=100, type=int, help='Number of episodes per epoch during test')

parser.add_argument('--batch_tasks', default=10, type=int, help='Number of mazes per batch during maml training')
parser.add_argument('--episodes_maml', default=20, type=int, help='Number of episodes per epoch during training in MAML')

parser.add_argument('--episode_per_update', default=50, type=int, help='Number of episodes before updating PPO') # 10

parser.add_argument('--adaptive_lr', default=False, type=bool, help='Per parameter adaptive learning rate in MAML')

parser.add_argument('--adaptation_trajectories', default=20, type=int, help='Number of trajectories used for the adaptation step (MAML), 0 means only one step')

parser.add_argument('--norm_A', default=1, type=int, help='normalize advantage in the outer loop for meta PPO')
parser.add_argument('--c1', default=0.5, type=float, help='scaling constant for value loss in PPO')
parser.add_argument('--c2', default=0.0, type=float, help='scaling constant for entropy bonus in PPO')
parser.add_argument('--gradient_clipping', default=1, type=int, help='clip gradients in PPO')
parser.add_argument('--eps_adapt', default=0.12, type=float, help='epsilon adaptation')

if __name__ == '__main__':
    args = parser.parse_args()
    params = args.__dict__

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # logdir = './logs_rl_project/agent=' + params['agent']
    #
    # if params['mbs'] > 1:
    #     logdir += '_multitask=' + str(params['mbs'])
    #
    # if params['maml']:
    #     logdir += '_MAML'
    #     params['episodes'] = params['episodes_maml']
    #     meta_learning(params, logdir, device)
    # else:
    #     multitask(params, logdir, device)




    params['agent'] = 'ppo'
    params['maml'] = True

    logdir = './logs_rl_project/agent='+params['agent']

    logdir += '_MAML_eps_adapt=' + str(params['eps_adapt'])
    params['episodes'] = params['episodes_maml']
    meta_learning(params, logdir, device)



    # for algo in ['dqn', 'ppo', 'reinforce']:
    #
    #     params['agent'] = algo
    #
    #     logdir = './logs_rl_project/agent='+params['agent']
    #
    #     if params['mbs'] > 1:
    #         logdir += '_multitask='+str(params['mbs'])
    #
    #     if params['maml']:
    #         logdir += '_MAML'
    #         params['episodes'] = params['episodes_maml']
    #         meta_learning(params, logdir, device)
    #     else:
    #         multitask(params, logdir, device)





    # if params['agent'] == 'dqn':
    #     if params['maml']:
    #         # meta_dqn(params, logdir, device)
    #         pass
    #     else:
    #         multitask(params, logdir, device)
    #
    # if params['agent'] == 'reinforce':
    #     if params['maml']:
    #         # meta_pg(params, logdir, device)
    #         pass
    #     else:
    #         multitask(params, logdir, device)
    #
    # if params['agent'] == 'ppo':
    #     if params['maml']:
    #         # meta_PPO(params, logdir, device)
    #         pass
    #     else:
    #         multitask(params, logdir, device)

    print()



