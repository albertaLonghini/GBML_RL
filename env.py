import numpy as np

"""
This class is the environment used to train the agent. The included functions are:
- step
- reset

"""
class Simulator:

    def __init__(self, start, goal, grid, T, params):
        """
        The initialization includes setting to POS_VAL the initial position of the grid (actually fixed to (1,1))
        and to GOAL_VAL the position of the goal (fixed to (20,20)).
        The idx parameter has been used to swap initial position and goal every even idx to increase
        the difficulty of the task while training multiple MDPs continuously.


        :param grid: Initial maze that has to be solved
        :type grid: nympy array Size(22,22)
        :param idx: number of the maze of the dataset we generated
        :type idx: int
        """
        self.init_pos = start             # FIXED by the dataset
        self.grid = grid                  # FIXED by the dataset
        self.goal = goal     # bcs mazes have the borders as 1 so we need a further step
        self.pos_val = params['pos_val']
        self.goal_val = params['goal_val']
        self.obs_val = params['obs_val']

        self.sparse = params['sparse_reward']

        self.show_goal = params['show_goal']

        self.T = T

        # self.grid[self.init_pos] = self.pos_val
        # self.grid[self.goal] = self.goal_val

        self.actual_pos_x = self.init_pos[0]
        self.actual_pos_y = self.init_pos[1]

    def step(self, a):
        """
        Step function that updates the grid given an action. Fistly updates the position on the grid based on the action
        taken and checks it.
        If the action brings to a position where there's an obstacle, then the agents stays in the previous position.
        The episodes end if the agent reaches the goal (or the maximum number of steps, but the last condition is
        evaluated during the training phase, not here)

        :param a: action to take
        :type a: int, either 0: down, 1: up, 2: right, 3: left
        :return r: reward after taking the action that is -l1_norm(position - goal)
        :rtype: float
        :return end: if the episode reached the end or not
        :rtype: bool
        """
        # Remove initial position
        # self.grid[self.actual_pos_x, self.actual_pos_y] = 0
        old_pos_x = self.actual_pos_x
        old_pos_y = self.actual_pos_y

        if a == 0:  # 0: forward
            self.actual_pos_x += 1
        elif a == 1:  # 1: backward
            self.actual_pos_x -= 1
        elif a == 2:  # 2: right
            self.actual_pos_y += 1
        elif a == 3:  # 3: left
            self.actual_pos_y -= 1

        # If hit obstacle: stay where you are
        if self.grid[self.actual_pos_x, self.actual_pos_y] == self.obs_val:
            # reset initial position without moving
            self.actual_pos_x = old_pos_x
            self.actual_pos_y = old_pos_y

        if self.sparse == 1:
            r = 0
        else:
            r = - (np.abs(self.actual_pos_x - self.goal[0]) + np.abs(self.actual_pos_y - self.goal[1])) / self.T  # todo: unnormalized and l1 norm
        # r = - (np.abs(self.actual_pos_x - self.goal[0])**2 + np.abs(self.actual_pos_y - self.goal[1])**2)

        # Check if goal reached
        if (self.actual_pos_x, self.actual_pos_y) == self.goal:
            return 1., False #True #TODO: back to True? exploration of multiple goals?

        # Set new position
        # self.grid[self.actual_pos_x, self.actual_pos_y] = self.pos_val
        return r, False

    def get_state(self):
        goal = np.zeros(self.grid.shape)
        goal[self.goal] = self.goal_val
        pos = np.zeros(self.grid.shape)
        pos[(self.actual_pos_x, self.actual_pos_y)] = self.pos_val
        if self.show_goal == 0:
            return np.expand_dims(np.concatenate([np.expand_dims(self.grid, 0), np.expand_dims(pos, 0)], 0), 0)
        else:
            return np.expand_dims(np.concatenate([np.expand_dims(self.grid, 0), np.expand_dims(pos, 0), np.expand_dims(goal, 0)], 0), 0)

    def reset(self):
        """
        Reset the environment to the initial state
        """
        # self.grid[self.actual_pos_x, self.actual_pos_y] = 0

        # self.grid[self.init_pos] = self.pos_val
        # self.grid[self.goal] = self.goal_val
        self.actual_pos_x = self.init_pos[0]
        self.actual_pos_y = self.init_pos[1]

    def normalize_reward(self, r, T):
        return r #- 1 + T * (T-1)/2.

    def get_distance(self):
        return - (np.abs(self.actual_pos_x - self.goal[0]) + np.abs(self.actual_pos_y - self.goal[1])) / self.T






