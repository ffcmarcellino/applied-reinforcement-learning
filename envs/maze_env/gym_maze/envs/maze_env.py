import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=False):

        self.viewer = None
        self.enable_render = enable_render

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640),
                                        enable_render=enable_render)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            self.maze_view.move_robot(action)

        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1
            done = True
        else:
            reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False

        self.state = self.maze_view.robot

        return self.state, reward, done, None, {}

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state, {}

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)

    def obs_to_state(self, obs):
        return int(self.maze_size[1]*obs[0] + obs[1])

    def get_dynamics(self):

        W, H = self.maze_size
        reward_dynamics = np.zeros((H*W, len(self.ACTION)))
        next_state_dynamics = np.zeros((H*W, len(self.ACTION), H*W))
        for i in range(W):
            for j in range(H):
                obs = np.array([i,j])
                for k in range(len(self.ACTION)):
                    a = self.ACTION[k]
                    next_obs = obs.copy()
                    if np.array_equal(obs, self.maze_view.goal):
                        reward_dynamics[self.obs_to_state(obs), k] = 0
                        next_state_dynamics[self.obs_to_state(obs), k, self.obs_to_state(next_obs)] = 1
                    else:
                        if self.maze_view.maze.is_open(obs, a):
                            next_obs += np.array(self.maze_view.maze.COMPASS[a])

                            if self.maze_view.maze.is_portal(next_obs):
                                next_obs = np.array(self.maze_view.maze.get_portal(tuple(next_obs)).teleport(tuple(next_obs)))
                        next_state_dynamics[self.obs_to_state(obs), k, self.obs_to_state(next_obs)] = 1
                        if np.array_equal(next_obs, self.maze_view.goal):
                            reward_dynamics[self.obs_to_state(obs), k] = 1
                        else:
                            reward_dynamics[self.obs_to_state(obs), k] = -0.1/(H*W)

        return next_state_dynamics, reward_dynamics


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), enable_render=enable_render)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", enable_render=enable_render)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), enable_render=enable_render)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", enable_render=enable_render)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), enable_render=enable_render)

class MazeEnvSample25x25(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvSample25x25, self).__init__(maze_file="maze2d_25x25.npy", enable_render=enable_render)

class MazeEnvSample30x30(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvSample30x30, self).__init__(maze_file="maze2d_30x30.npy", enable_render=enable_render)

class MazeEnvSample50x50(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvSample50x50, self).__init__(maze_file="maze2d_50x50.npy", enable_render=enable_render)

class MazeEnvSample100x100(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", enable_render=enable_render)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), enable_render=enable_render)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", enable_render=enable_render)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, enable_render=False):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", enable_render=enable_render)
