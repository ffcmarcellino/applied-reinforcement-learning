import numpy as np
from tqdm import tqdm

class MDP:

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def reset(self):
        init_state, _ = self.env.reset()
        init_action = self.agent.reset(init_state)
        return init_state, init_action, False

    def step(self, state, action):
        next_state, reward, is_terminal, _ = self.env.step(state, action)
        next_action = self.agent.step(next_state, reward)
        return reward, next_state, next_action, is_terminal

class RLTask(MDP):

    def __init__(self, agent, env, max_episode_lenght=10e6):
        super().__init__(agent, env)
        self.max_episode_lenght = max_episode_lenght

    def run_episode(self):
        state, action, is_terminal = self.reset()
        for i in range(self.max_episode_lenght):
            state, action, is_terminal = self.step(state, action)
            if is_terminal:
                break

    def run_mdp(self, num_iterations, metrics_info=None):

        metrics = {metric: [] for metric in metrics_info}
        state, action, _ = self.reset()

        for i in range(num_iterations):

            if 'optimal_action_flag' in metrics_info:
                self._update_optimal_action_flag(metrics['optimal_action_flag'], action, metrics_info['optimal_action_flag']['optimal_action'])

            reward, state, action, _ = self.step(state, action)

            if 'cum_avg_reward' in metrics_info:
                self._update_cum_avg_reward(metrics['cum_avg_reward'], reward)

        if 'final_policy' in metrics_info:
            self._update_final_policy(metrics, metrics_info['final_policy']['states'])

        return metrics

    def run_independent_mdps(self, num_runs, num_iterations, metrics_info=None):

        metrics = self._initialize_metrics(num_iterations, metrics_info)

        for i in tqdm(range(num_runs)):
            state, action, _ = self.reset()
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
