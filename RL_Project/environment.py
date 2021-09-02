import numpy as np


def step(x, y, direction, grid):
    grid_shape = grid.shape
    neighbor = (x + direction[0], y + direction[1])
    valid = 0 <= neighbor[0] < grid_shape[0] and 0 <= neighbor[1] < grid_shape[1]
    no_wall = valid and grid[neighbor] == 0

    return no_wall, neighbor


def BFS(lvl_queue, queue, grid, visited, count, goal):
    if len(queue) == 0:
        return False, count, lvl_queue

    next_queue = []
    for node in queue:
        directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        for direction in directions:
            valid, neighbor = step(node[0], node[1], direction, grid)

            if valid and not visited[neighbor]:
                if neighbor == goal:
                    lvl_queue.append([goal])
                    return True, count + 1, lvl_queue

                visited[neighbor] = True
                next_queue.append(neighbor)

    lvl_queue.append(next_queue)
    return BFS(lvl_queue, next_queue, grid, visited, count + 1, goal)


class Simulator:

    def __init__(self, params, grid=None, T=None):

        if grid is None:
            grid, T = self.get_maze()

        self.init_pos = (1, 1)             # FIXED by the dataset
        self.grid = grid                  # FIXED by the dataset
        self.goal = (10, 10)     # bcs mazes have the borders as 1 so we need a further step
        self.T = T
        self.pos_val = 1
        self.goal_val = 1
        self.obs_val = 1

        # self.grid[self.init_pos] = self.pos_val
        # self.grid[self.goal] = self.goal_val

        self.actual_pos_x = self.init_pos[0]
        self.actual_pos_y = self.init_pos[1]

    def step(self, a):

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

        r = - (np.abs(self.actual_pos_x - self.goal[0]) + np.abs(self.actual_pos_y - self.goal[1])) / self.T # todo: unnormalized and l1 norm
        # r = - (np.abs(self.actual_pos_x - self.goal[0])**2 + np.abs(self.actual_pos_y - self.goal[1])**2)

        # Check if goal reached
        if (self.actual_pos_x, self.actual_pos_y) == self.goal:
            return 1., True

        # Set new position
        # self.grid[self.actual_pos_x, self.actual_pos_y] = self.pos_val
        return r, False

    def get_state(self):
        goal = np.zeros(self.grid.shape)
        goal[self.goal] = self.goal_val
        pos = np.zeros(self.grid.shape)
        pos[(self.actual_pos_x, self.actual_pos_y)] = self.pos_val
        return np.expand_dims(np.concatenate([np.expand_dims(self.grid, 0), np.expand_dims(pos, 0), np.expand_dims(goal, 0)], 0), 0)

    def reset(self):
        self.actual_pos_x = self.init_pos[0]
        self.actual_pos_y = self.init_pos[1]

    # def normalize_reward(self, r, T):
    #     return r - 1 + T * (T-1)/2.

    def get_maze(self):

        while True:
            temp = np.random.binomial(1, 0.2, (10, 10))
            grid = np.ones((temp.shape[0] + 2, temp.shape[0] + 2))
            grid[1:-1, 1:-1] = temp

            start = (1, 1)
            goal = (10, 10)
            grid[start] = grid[goal] = 0

            queue = []
            lvl_queue = []
            queue.append(start)
            lvl_queue.append([start])

            visited = np.full(grid.shape, False, dtype=bool)
            visited[start] = True
            count = 0
            BFS_res, BFS_count, nodes = BFS(lvl_queue, queue, grid, visited, count, goal)

            if BFS_res:
                return grid.copy(), BFS_count






