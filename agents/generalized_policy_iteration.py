import numpy as np
from .initializers import Initializer
from .action_selection import greedy_policy

class PolicyIteration:

    def __init__(self, num_states, num_actions, initializer_params, terminal_states, theta, reward_dynamics, next_state_dynamics, discount_factor=1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.state_values = None
        self.policy = None
        self.initializer_params = initializer_params
        self.initializer = Initializer(num_states)
        self.terminal_states = terminal_states
        self.theta = theta
        self.reward_dynamics = reward_dynamics
        self.next_state_dynamics = next_state_dynamics
        self.discount_factor = discount_factor

    def reset(self):
        self.state_values = self.initializer.initialize(**self.initializer_params)
        self.state_values[self.terminal_states] = 0
        self.policy = Initializer.constant_initializer(constant=1./self.num_actions, dims=(self.num_states, self.num_actions))

    def policy_evaluation(self, max_iterations=10**6, metrics_info=None):

        metrics = {metric: [] for metric in metrics_info} if metrics_info is not None else {}

        for i in range(max_iterations):
            action_values = self.reward_dynamics + self.discount_factor * np.sum(self.state_values.reshape((1,1,-1))*self.next_state_dynamics, axis=2)
            state_values = (self.policy * action_values).sum(axis=1)
            if 'rmse' in metrics:
                metrics['rmse'].append(np.mean((self.state_values - metrics_info['rmse']['target_state_values'])**2)**(0.5))
            if 'max_abs_err' in metrics:
                metrics['max_abs_err'].append(np.max(np.abs(self.state_values - metrics_info['max_abs_err']['target_state_values'])))
            delta = np.abs(state_values - self.state_values).max()
            self.state_values = state_values
            if delta < self.theta:
                break
        if 'num_iterations' in metrics:
            metrics['num_iterations'] = [i+1]
        return i+1, metrics

    def policy_improvement(self):

        action_values = self.reward_dynamics + self.discount_factor * np.sum(self.state_values.reshape((1,1,-1))*self.next_state_dynamics, axis=2)
        self.policy = greedy_policy(action_values, tie_break='random')

    def run(self, max_iterations=10**6, max_eval_iterations=10**6, metrics_info=None):

        metrics = {metric: [] for metric in metrics_info} if metrics_info is not None else {}

        for i in range(max_iterations):
            last_state_values = self.state_values.copy()
            _, i_metrics = self.policy_evaluation(max_iterations=max_eval_iterations, metrics_info=metrics_info)
            for metric in metrics:
                metrics[metric] += i_metrics[metric]
            self.policy_improvement()
            delta = np.abs(last_state_values - self.state_values).max()
            if delta < self.theta:
                break

        return metrics

class ValueIteration(PolicyIteration):
    def run(self, max_iterations=10**6, target_state_values=None, metrics_info=None):
        return super().run(max_iterations=max_iterations, max_eval_iterations=1, metrics_info=metrics_info)
