import numpy as np
from tqdm import tqdm
import time

class MDP:

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def start(self):
        init_state, _ = self.env.reset()
        init_action = self.agent.start(init_state)
        return init_state, init_action

    def step(self, state, action):
        next_state, reward, is_terminal, is_truncated, info = self.env.step(action)
        next_action = self.agent.step(next_state, reward)
        return reward, next_state, next_action, is_terminal, is_truncated, info

class RLTask(MDP):

    def __init__(self, agent, env, max_episode_length=10**6):
        super().__init__(agent, env)
        self.max_episode_length = max_episode_length

    def run_episode(self, metric_objs=None, render_lag=None):

        metrics = {metric.NAME: [] for metric in metric_objs} if metric_objs is not None else {}
        state, action = self.start()

        if render_lag is not None:
            self.env.render()

        for i in range(self.max_episode_length):
            reward, state, action, is_terminal, is_truncated, info = self.step(state, action)
            self._compute_metrics(metrics, metric_objs, 'on_episode_step', state=state, action=action, reward=reward, agent=self.agent, step=i+1, info=info)
            if render_lag is not None:
                time.sleep(render_lag)
                self.env.render()
            if is_terminal or is_truncated:
                self.agent.on_episode_end()
                break

        if (not is_truncated) and (not is_terminal):
            self.agent.on_episode_end()
        
        self._compute_metrics(metrics, metric_objs, 'on_episode_end', reward=reward, agent=self.agent, step=i+1)

        return metrics
    
    def test_episode(self, metric_objs=None, test_env=None, render_lag=None, save=False):
        
        test_env = test_env or self.env
    
        metrics = {metric.NAME: [] for metric in metric_objs} if metric_objs is not None else {}

        state, _ = test_env.reset()
    
        if render_lag is not None:
            time.sleep(render_lag)
            test_env.render(save=save)
    
        action = self.agent.select_action(state, greedy=True)
    
        self._compute_metrics(metrics, metric_objs, 'on_episode_test_start', state=state, action=action)
        
        for i in range(self.max_episode_length):
            state, reward, is_terminal, is_truncated, info = test_env.step(action)
            if render_lag is not None: 
                time.sleep(render_lag)
                test_env.render(save=save)
            action = self.agent.select_action(state, greedy=True)
            self._compute_metrics(metrics, metric_objs, 'on_episode_test_step', state=state, action=action, reward=reward, step=i+1, info=info)
            if is_terminal or is_truncated:
                break

        self._compute_metrics(metrics, metric_objs, 'on_episode_test_end', reward=reward, step=i+1)
            
        return metrics

    def run_mdp(self, num_iterations, metric_objs=None, render_lag=None):

        metrics = {metric.NAME: [] for metric in metric_objs} if metric_objs is not None else {}
        state, action = self.start()

        for i in range(num_iterations):

            self._compute_metrics(metrics, metric_objs, 'on_iteration_start', action=action)

            reward, state, action, _, _, info = self.step(state, action)

            self._compute_metrics(metrics, metric_objs, 'on_iteration_end', reward=reward, step=i+1, info=info)

        self._compute_metrics(metrics, metric_objs, 'on_run_end', agent=self.agent)
      
        return metrics

    def run_independent_mdps(self, num_runs, num_iterations, metric_objs=None):

        metrics = self._initialize_metrics(num_iterations, metric_objs)

        for i in tqdm(range(num_runs)):
            self.agent.reset()
            metrics_i = self.run_mdp(num_iterations, metric_objs)
            for metric in metrics:
                metrics[metric] += (np.array(metrics_i[metric])*1 - metrics[metric])/(i+1)

        return metrics
    
    def _compute_metrics(self, metrics, metric_objs, location, **kwargs):
        if metric_objs is not None:
            for metric in metric_objs:
                if metric.location == location:
                    metric.update(metrics, **kwargs)

    def _initialize_metrics(self, num_iterations, metric_objs):

        if metric_objs is None:
            return {}

        metrics = {metric.NAME: np.zeros(num_iterations) for metric in metric_objs}

        for metric in metric_objs:
            if metric.NAME == 'greedy_policy':
                dims = self.agent.num_actions if type(metric.states) == int else (len(metric.states),self.agent.num_actions)
                metrics[metric.NAME] = np.zeros(dims)

        return metrics