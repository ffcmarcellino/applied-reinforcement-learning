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

    def policy_evaluation(self, max_iterations=10**6):
        for i in range(max_iterations):
            action_values = self.reward_dynamics + self.discount_factor * np.sum(self.state_values.reshape((1,1,-1))*self.next_state_dynamics, axis=2)
            state_values = (self.policy * action_values).sum(axis=1)
            delta = np.abs(state_values - self.state_values).max()
            if delta < self.theta:
                break
            self.state_values = state_values

    def policy_improvement(self):
        action_values = self.reward_dynamics + self.discount_factor * np.sum(self.state_values.reshape((1,1,-1))*self.next_state_dynamics, axis=2)
        self.policy = greedy_policy(action_values, tie_break='random')

    def run(self, max_iterations=10**6, max_eval_iterations=10**6):
        for i in range(max_iterations):
            last_state_values = self.state_values.copy()
            self.policy_evaluation(max_iterations=max_eval_iterations)
            self.policy_improvement()
            delta = np.abs(last_state_values - self.state_values).max()
            if delta < self.theta:
                break

class ValueIteration(PolicyIteration):
    def run(self, max_iterations=10**6):
        super().run(max_iterations=max_iterations, max_eval_iterations=1)
