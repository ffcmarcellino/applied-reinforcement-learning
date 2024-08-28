from gymnasium.envs.toy_text import TaxiEnv as TEnv
import imageio

class TaxiEnv(TEnv):

    def __init__(self, render_mode = None, seed = None, save = False):
        super().__init__(render_mode)
        self.grids=[]
        self.seed = seed
    
    def reset(self, seed = None):
        env_seed = self.seed or seed
        return super().reset(seed=env_seed)

    def render(self, save=False):
        img = super().render()
        if save:
            self.grids.append(img)

    def set_seed(self, seed):
        self.seed = seed
        
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