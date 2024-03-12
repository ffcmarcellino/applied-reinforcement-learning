from typing import Any, SupportsFloat
import imageio
import numpy as np

from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, Door, Floor
from minigrid.minigrid_env import MiniGridEnv


class LavaEnv(MiniGridEnv):

    DEFAULT_REWARD = -1
    GOAL_REWARD = 100
    LAVA_REWARD = -500

    def __init__(
        self,
        max_steps=None,
        **kwargs,
    ):

        self.grids = []
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * 12**2

        super().__init__(
            mission_space=mission_space,
            grid_size=12,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Lava Mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate Lava
        for i in range(2, 8, 2):
            for j in range(4, 10, 2):
                self.grid.set(i, j, Lava())

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), 3, 1)

        # Place the agent
        self.agent_pos = (3, 10)
        self.agent_dir = 3

        self.mission = self._gen_mission()

    def step(
        self, action: int
    ):
        self.step_count += 1

        reward = self.DEFAULT_REWARD
        terminated = False
        truncated = False

        # Go UP
        if action == 3:
            self.agent_dir = 3

        # Go DOWN
        elif action == 1:
            self.agent_dir = 1

        # Go LEFT
        elif action == 2:
            self.agent_dir = 2

        # Go RIGHT
        elif action == 0:
            self.agent_dir = 0

        else:
            raise ValueError(f"Unknown action: {action}")

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)
        if fwd_cell is not None and fwd_cell.type == "goal":
            terminated = True
            reward = self.GOAL_REWARD
        if fwd_cell is not None and fwd_cell.type == "lava":
            terminated = True
            reward = self.LAVA_REWARD

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def render(self, save=False):
        if save:
            img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
            self.grids.append(img)
        super().render()

    def save_gif(self, path, duration):
        """
        Saves grid image arrays into a GIF file.

        Args:
            str path: path to save GIF file.
            int duration: duration of each frame in ms
        """

        imageio.mimsave(path, self.grids, duration=duration, loop=0)

    def clean_grids(self):
        """
        Empties list of grid image arrays.
        """

        self.grids = []

    def gen_obs(self):
        x = self.agent_pos[0]-1
        y = self.agent_pos[1]-1
        return y*(self.width-2) + x

class MultiGoalEnv(MiniGridEnv):

    DEFAULT_REWARD = -0.1
    DOOR_GOAL_REWARD = 120
    GOAL_REWARD = 100
    LAVA_GOAL_REWARD = 200
    LAVA_REWARD = -2000
    STOCHASTIC_LAVA_REWARD = (-2, 5)

    LAVA_Y_COORD = 2
    WALL_COORD = (10,8)
    START_POS = (6,11)
    STOCHASTIC_LAVA_POS = (5,11)

    DOOR_GOAL_POS = (1,6)
    GOAL_POS = (11,11)
    LAVA_GOAL_POS = (6,1)

    def __init__(
        self,
        max_steps=None,
        stochastic=False,
        save=False,
        **kwargs,
    ):

        self.grids = []
        self.stochastic=stochastic
        self.save = save
        mission_space = MissionSpace(mission_func=self._gen_mission)
        grid_size = 13

        if max_steps is None:
            max_steps = 100 * grid_size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Multiple Goals"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical wall
        self.grid.vert_wall(*self.WALL_COORD)

        # Generate room
        self.grid.wall_rect(self.DOOR_GOAL_POS[0]-1, self.DOOR_GOAL_POS[1]-1, 3, 3)

        # Place a door in the wall
        self.put_obj(Door("yellow", is_locked=False), self.DOOR_GOAL_POS[0]+1, self.DOOR_GOAL_POS[1])

        # Generate Lava
        for i in range(1, self.width//2):
            self.grid.set(i, self.LAVA_Y_COORD, Lava())
        for i in range(self.width//2+1, self.width-1):
            self.grid.set(i, self.LAVA_Y_COORD, Lava())
        if self.stochastic:
            self.grid.set(self.STOCHASTIC_LAVA_POS[0], self.STOCHASTIC_LAVA_POS[1], Lava())

        # Place goals
        for goal_pos in (self.DOOR_GOAL_POS, self.GOAL_POS, self.LAVA_GOAL_POS):
            self.put_obj(Goal(), *goal_pos)

        # Place the agent
        self.agent_pos = self.START_POS
        self.agent_dir = 3

        self.mission = self._gen_mission()

    def step(
        self, action: int
    ):
        self.step_count += 1

        reward = self.DEFAULT_REWARD
        terminated = False
        truncated = False

        # Get the contents of the cell in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Move
        if action >= 0 and action <= 3:

            # Turn the agent's direction
            self.agent_dir = action

            # Get the contents of the cell in front of the agent
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            # Move forward
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                if self.agent_pos == self.GOAL_POS:
                    reward = self.GOAL_REWARD
                elif self.agent_pos == self.DOOR_GOAL_POS:
                    reward = self.DOOR_GOAL_REWARD
                elif self.agent_pos == self.LAVA_GOAL_POS:
                    reward = self.LAVA_GOAL_REWARD
            elif fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True
                reward = np.random.normal(self.STOCHASTIC_LAVA_REWARD[0], self.STOCHASTIC_LAVA_REWARD[1]) if self.agent_pos == self.STOCHASTIC_LAVA_POS else self.LAVA_REWARD

        # Toggle/activate an object
        elif action == 4:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render(save=self.save)

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def render(self, save=False):
        if save:
            img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
            self.grids.append(img)
        super().render()

    def save_gif(self, path, duration):
        """
        Saves grid image arrays into a GIF file.

        Args:
            str path: path to save GIF file.
            int duration: duration of each frame in ms
        """

        imageio.mimsave(path, self.grids, duration=duration, loop=0)

    def clean_grids(self):
        """
        Empties list of grid image arrays.
        """

        self.grids = []

    def obs_to_state(self, agent_pos, door_state, agent_dir):

        x = agent_pos[0]-1
        y = agent_pos[1]-1

        sub_state = (y*(self.width-2) + x)*2 + door_state

        if agent_pos == (self.DOOR_GOAL_POS[0]+2, self.DOOR_GOAL_POS[1]):
            return sub_state if agent_dir != 2 else (self.width-2)*(self.height-2)*2 + door_state

        return sub_state

    def gen_obs(self):
        door_state = self.grid.get(self.DOOR_GOAL_POS[0]+1, self.DOOR_GOAL_POS[1]).is_open*1
        return self.obs_to_state(self.agent_pos, door_state, self.agent_dir)
