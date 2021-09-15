import numpy as np


def step(x, y, direction, grid):
    grid_shape = grid.shape
    neighbor = (x + direction[0], y + direction[1])
    valid = 0 <= neighbor[0] < grid_shape[0] and 0 <= neighbor[1] < grid_shape[1]
    no_wall = valid and grid[neighbor] == 0

    return no_wall, neighbor



def DFS(nodes, level, paths, tmp_path, current_node, start_node):

    if current_node == start_node:
        tmp_path = tmp_path.copy()
        tmp_path.reverse()
        paths.append(tmp_path)
        return paths

    for node in nodes[level+1]:
        if abs(node[0]-current_node[0])+abs(node[1]-current_node[1]) == 1:
            tmp_path.append(node)
            paths = DFS(nodes, level+1, paths, tmp_path, node, start_node)
            del tmp_path[-1]

    return paths


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


class Maze_Gen:

    def __init__(self, params):
        self.size = params['grid_size']
        self.path_length = params['path_length']
        self.mods = params['modalities_goal_dist']
        self.p = params['p_obstacles']

    def get_maze(self, central=True):
        while True:
            temp = np.random.binomial(1, self.p, (self.size, self.size))
            grid = np.ones((temp.shape[0] + 2, temp.shape[0] + 2))
            grid[1:-1, 1:-1] = temp

            if central:
                start = (int(self.size / 2) + 1, int(self.size / 2) + 1)
            else:
                start = (np.random.randint(0, self.size) + 1, np.random.randint(0, self.size) + 1)
            goal = self.place_goal(np.array(start))
            grid[start] = grid[goal] = 0

            queue = []
            lvl_queue = []
            queue.append(start)
            lvl_queue.append([start])

            visited = np.full(grid.shape, False, dtype=bool)
            visited[start] = True
            count = 0
            BFS_res, BFS_count, nodes = BFS(lvl_queue, queue, grid, visited, count, goal)

            if BFS_res:  # and BFS_count == self.path_length:
                tmp_path = [goal]
                nodes.reverse()
                paths = DFS(nodes, 0, [], tmp_path, goal, start)
                return start, goal, grid.copy(), BFS_count, self.get_action(paths)

    def place_goal(self, starting_point):
        T = self.path_length

        if self.mods == 0:
            while True:
                quadrant = np.random.randint(0, 4)
                i = np.random.randint(0, T)
                goals = np.array([[- T + i, i], [i, T - i], [T - i, - i], [- i, - T + i]])
                goal = goals[quadrant] + starting_point

                if 0 < goal[0] <= self.size and 0 < goal[1] <= self.size:
                    return tuple(goal)
        else:
            while True:
                quadrant = np.random.randint(0, np.minimum(self.mods, 2))
                i_s = [np.random.randint(1, np.clip(int(self.mods / 4) + np.clip(self.mods % 4 - 0, 0, 1), 1, T)),
                       np.random.randint(0, np.clip(int(self.mods / 4) + np.clip(self.mods % 4 - 1, 0, 1), 1, T)),
                       np.random.randint(0, np.clip(int(self.mods / 4) + np.clip(self.mods % 4 - 2, 0, 1), 1, T)),
                       np.random.randint(0, np.clip(int(self.mods / 4) + np.clip(self.mods % 4 - 3, 0, 1), 1, T))]
                goals = np.array([[- T + i_s[0], i_s[0]], [i_s[1], T - i_s[1]], [T - i_s[2], - i_s[2]], [- i_s[3], - T + i_s[3]]])
                goal = goals[quadrant] + starting_point
                if 0 < goal[0] <= self.size and 0 < goal[1] <= self.size:
                    return tuple(goal)

    def get_action(self, paths):
        actions = []
        for path in paths:
            temp_actions = []
            current_state = path[0]
            for i in range(1, len(path)):
                if path[i][0] - current_state[0] == 1:
                    temp_actions.append(0)
                if path[i][0] - current_state[0] == -1:
                    temp_actions.append(1)
                if path[i][1] - current_state[1] == 1:
                    temp_actions.append(2)
                if path[i][1] - current_state[1] == -1:
                    temp_actions.append(3)
                current_state = path[i]
            actions.append(temp_actions)

        return actions