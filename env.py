import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import heapq


class RobustEnv(gym.Env):
    def __init__(self):
        super(RobustEnv, self).__init__()
        self.grid_size = 9  # 地图大小
        self.max_steps = 200  # 最大步数限制
        self.action_space = spaces.Discrete(4)  # 动作空间：上下左右四个方向
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.float32)  # 观察空间：代理的坐标
        self.agent_pos = np.array([0, 0])  # 代理起始位置
        self.goal_pos = np.array([8, 8])  # 目标位置
        self.obstacle_pos = self._generate_obstacles()  # 障碍物位置
        self.current_step = 0  # 当前步数
        self.prev_pos = None  # 记录代理之前的位置
        self.blocked_positions = set(self.obstacle_pos)  # 初始化为障碍物位置

    def _generate_obstacles(self):
        obstacles = set()
        num_obstacles = 14  # 障碍物数量

        def is_path_blocked(start, end, obstacles):
            visited = set()
            queue = deque([start])
            while queue:
                current = queue.popleft()
                if current == end:
                    return False
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    next_pos = (current[0] + dx, current[1] + dy)
                    if (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[
                        1] < self.grid_size and next_pos not in visited and next_pos not in obstacles):
                        visited.add(next_pos)
                        queue.append(next_pos)
            return True

        while len(obstacles) < num_obstacles:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if pos != tuple(self.agent_pos) and pos != tuple(self.goal_pos):
                obstacles.add(pos)
                if is_path_blocked(tuple(self.agent_pos), tuple(self.goal_pos), obstacles):
                    obstacles.remove(pos)
        return obstacles

    def _heuristic(self, a, b):
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    def _a_star_search(self, start, goal):
        Node = namedtuple('Node', ['position', 'cost', 'priority'])
        open_list = []
        heapq.heappush(open_list, Node(start, 0, 0))
        came_from = {}
        cost_so_far = {start: 0}

        while open_list:
            current = heapq.heappop(open_list).position

            if current == goal:
                break

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if 0 <= next_pos[0] < self.grid_size and 0 <= next_pos[
                    1] < self.grid_size and next_pos not in self.blocked_positions:
                    new_cost = cost_so_far[current] + 1
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + self._heuristic(next_pos, goal)
                        heapq.heappush(open_list, Node(next_pos, new_cost, priority))
                        came_from[next_pos] = current

        # Reconstruct path
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agent_pos = np.array([0, 0])
        self.current_step = 0
        self.prev_pos = self.agent_pos.copy()
        self.obstacle_pos = self._generate_obstacles()
        self.blocked_positions = set(self.obstacle_pos)
        return self.agent_pos, {}

    def step(self, action):
        def distance(pos1, pos2):
            return np.sqrt(np.sum((pos1 - pos2) ** 2))

        path = self._a_star_search(tuple(self.agent_pos), tuple(self.goal_pos))
        if len(path) > 1:
            next_pos = path[1]
        else:
            next_pos = path[0]

        self.prev_pos = self.agent_pos.copy()
        self.agent_pos = np.array(next_pos)

        reward = 1 / (distance(self.agent_pos, self.goal_pos) + 1e-5) - self.current_step / self.max_steps
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 10

        self.current_step += 1
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.current_step >= self.max_steps

        return self.agent_pos, reward, done, {}, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.agent_pos[1], self.agent_pos[0]] = 1
        grid[self.goal_pos[1], self.goal_pos[0]] = 2
        for pos in self.obstacle_pos:
            grid[pos[1], pos[0]] = -1
        plt.imshow(grid, cmap='viridis', interpolation='none')
        plt.pause(0.1)
        plt.clf()