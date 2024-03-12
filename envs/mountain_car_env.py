from typing import Optional
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

class DiscreteMountainCarEnv(MountainCarEnv):
    
    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0, grid_size=10, seed=0):
        super().__init__(render_mode, goal_velocity)
        self.grid_size = grid_size
        self.seed = seed
    
    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)
        return self._obs_to_state(obs), reward, terminated, truncated, {'obs': obs}
    
    def reset(self, *args, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(*args, seed=seed or self.seed, options=options)
        self.seed += 1
        return self._obs_to_state(obs), info
    
    def render(self, save=False):
        super().render()
    
    def state_to_obs(self, state):
        
        p = state % self.grid_size
        min_p = p * (self.max_position - self.min_position) / self.grid_size + self.min_position
        max_p = (p+1) * (self.max_position - self.min_position) / self.grid_size + self.min_position
        
        v = state // self.grid_size
        min_v = 2 * v * self.max_speed / self.grid_size - self.max_speed
        max_v = 2 * (v+1) * self.max_speed / self.grid_size - self.max_speed

        return ((min_p, max_p), (min_v, max_v))
    
    def _obs_to_state(self, obs):
        
        p = (obs[0]-self.min_position) * self.grid_size / (self.max_position - self.min_position)
        p = int(p) if p < self.grid_size else self.grid_size - 1
        
        v = (obs[1] + self.max_speed) * self.grid_size / (2*self.max_speed)
        v = int(v) if v < self.grid_size else self.grid_size - 1
        
        return v*self.grid_size + p
    

    
    