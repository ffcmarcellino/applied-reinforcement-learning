import numpy as np
from tqdm import tqdm
import time

class MDP:

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def reset(self):
        init_state, _ = self.env.reset()
        init_action = self.agent.reset(init_state)
        return init_state, init_action

    def step(self, state, action):
        next_state, reward, is_terminal, is_truncated, _ = self.env.step(action)
        next_action = self.agent.step(next_state, reward)
        return reward, next_state, next_action, is_terminal, is_truncated

class RLTask(MDP):

    def __init__(self, agent, env, max_episode_length=10**6):
        super().__init__(agent, env)
        self.max_episode_length = max_episode_length

    def run_episode(self, metrics_info=None, render_lag=None):

        metrics = {metric: [] for metric in metrics_info} if metrics_info is not None else {}
        state, action = self.reset()
        if render_lag is not None:
            self.env.render()
        for i in range(self.max_episode_length):
            reward, state, action, is_terminal, is_truncated = self.step(state, action)
            if render_lag is not None:
                time.sleep(render_lag)
                self.env.render()
            if 'cum_reward' in metrics:
                self._update_cum_reward(metrics, reward)
            if is_terminal or is_truncated:
                self.agent.on_episode_end()
                break

        if 'num_steps' in metrics:
            metrics['num_steps'] = self.agent.t

        if 'perc_pair_visits' in metrics:
            num_pairs = np.prod(self.agent.action_values.shape) if self.agent.allowed_actions_mask is None else self.agent.allowed_actions_mask.sum()
            metrics['perc_pair_visits'] = (self.agent.num_visits != 0).sum() / num_pairs

        return metrics

    def run_mdp(self, num_iterations, metrics_info=None, render_lag=None):

        metrics = {metric: [] for metric in metrics_info} if metrics_info is not None else {}
        state, action = self.reset()

        for i in range(num_iterations):

            if 'optimal_action_flag' in metrics:
                self._update_optimal_action_flag(metrics['optimal_action_flag'], action, metrics_info['optimal_action_flag']['optimal_action'])

            reward, state, action, _, _ = self.step(state, action)

            if 'cum_avg_reward' in metrics:
                self._update_cum_avg_reward(metrics['cum_avg_reward'], reward)

        if 'final_policy' in metrics:
            self._update_final_policy(metrics, metrics_info['final_policy']['states'])

        return metrics

    def run_independent_mdps(self, num_runs, num_iterations, metrics_info=None):

        metrics = self._initialize_metrics(num_iterations, metrics_info)

        for i in tqdm(range(num_runs)):
            metrics_i = self.run_mdp(num_iterations, metrics_info)
            for metric in metrics:
                metrics[metric] += (np.array(metrics_i[metric])*1 - metrics[metric])/(i+1)

        return metrics

    def _initialize_metrics(self, num_iterations, metrics_info):

        if metrics_info is None:
            return {}

        metrics = {metric: np.zeros(num_iterations) for metric in metrics_info}

        if 'final_policy' in metrics_info:
            dims = self.agent.num_actions if type(metrics_info['final_policy']['states']) == int else (len(metrics_info['final_policy']['states']),self.agent.num_actions)
            metrics['final_policy'] = np.zeros(dims)

        return metrics

    def _update_optimal_action_flag(self, optimal_action_flag, action, optimal_action):
        optimal_action_flag.append((action==optimal_action)*100)

    def _update_cum_avg_reward(self, cum_avg_reward, reward):
        avg_reward = 0 if self.agent.t == 1 else cum_avg_reward[-1]
        avg_reward += (reward - avg_reward)/self.agent.t
        cum_avg_reward.append(avg_reward)

    def _update_final_policy(self, metrics, states):
        metrics['final_policy'] = self.agent.get_policy(states)

    def _update_cum_reward(self, metrics, reward):
        metrics['cum_reward'].append(reward)
        metrics['cum_reward'] = [sum(metrics['cum_reward'])]
