from .action_selection import *
from .initializers import Initializer

class BaseAgent:

    def __init__(self, num_states, num_actions, agent_info):
        self.num_states = num_states
        self.num_actions = num_actions
        self.initializer = Initializer(self.num_states, self.num_actions)
        self.initializer_params = agent_info['initializer_params']
        self.get_policy = self._get_policy_method(agent_info['action_selection_params']['method'], **agent_info['action_selection_params'].get('kwargs', {}))
        self._step_size = self._get_step_size_fun(agent_info['step_size'])

        self.t = None
        self.num_visits = None
        self.action_values = None
        self.last_state = None
        self.last_action = None

    def reset(self, init_state):
        self.t = 0
        self.num_visits = self.initializer.initialize("zero")
        self.action_values = self.initializer.initialize(**self.initializer_params)
        self.last_state = init_state
        self.last_action = self.select_action(init_state)
        return self.last_action

    def select_action(self, state):
        policy = self.get_policy(state)
        if isinstance(policy, np.int64):
            return policy
        else:
            return policy_action_selection(policy)

    def get_greedy_policy(self, states, **greedy_policy_kwargs):
        return greedy_policy(self.action_values[states], **greedy_policy_kwargs)

    def get_epsilon_greedy_policy(self, states, **epsilon_greedy_policy_kwargs):
        return epsilon_greedy_policy(self.action_values[states], **epsilon_greedy_policy_kwargs)

    def get_ucb_policy(self, states, **ucb_policy_kwargs):
        return ucb_policy(self.action_values[states], self.num_visits[states], **ucb_policy_kwargs)

    def get_softmax_policy(self, states):
        return softmax_policy(self.action_values[states])

    def _inverse_count(self, state=None, action=None):

        if state is None:
            return 1/self.t
        else:
            return 1/self.num_visits[state].sum() if action is None else 1/self.num_visits[state, action]

    def _get_step_size_fun(self, step_size):

        if type(step_size) in (int, float):
            def step_size_fun(s,a):
                return step_size

        elif step_size == '1/n':
            return self._inverse_count

        else:
            raise NameError(f"{step_size} is not a valid action step size function.")

        return step_size_fun

    def _get_policy_method(self, action_selection_method, **action_selection_kwargs):

        if action_selection_method == 'greedy':
            def get_policy(state):
                 return self.get_greedy_policy(state, **action_selection_kwargs)

        elif action_selection_method == 'epsilon_greedy':
            def get_policy(state):
                 return self.get_epsilon_greedy_policy(state, **action_selection_kwargs)

        elif action_selection_method == 'ucb':
            def get_policy(state):
                 return self.get_ucb_policy(state, **action_selection_kwargs)

        elif action_selection_method == 'softmax':
            def get_policy(state):
                 return self.get_softmax_policy(state, **action_selection_kwargs)

        else:
            raise(f"{action_selection_params['method']} is not a valid action selection method.")

        return get_policy

class KBandits(BaseAgent):

    def step(self, state, reward):
        self.t += 1
        self.num_visits[self.last_state, self.last_action] += 1
        self.action_values[self.last_state, self.last_action] += self._step_size(self.last_state, self.last_action)*(reward - self.action_values[self.last_state, self.last_action])
        self.last_state = state
        self.last_action = self.select_action(state)
        return self.last_action

class GradientBandits(BaseAgent):

    def __init__(self, num_states, num_actions, agent_info):
        agent_info['action_selection_params'] = {'method': 'softmax'}
        super().__init__(num_states, num_actions, agent_info)
        self.avg_reward = None

    def reset(self, init_state):
        super().reset(init_state)
        self.avg_reward = self.initializer.zero_initializer(self.num_states)

    def step(self, state, reward):
        self.t += 1
        self.num_visits[self.last_state, self.last_action] += 1
        self.action_values[self.last_state] -= self._step_size(self.last_state, self.last_action)*(reward - self.avg_reward[self.last_state])*self.get_policy(self.last_state)
        self.action_values[self.last_state, self.last_action] += self._step_size(self.last_state, self.last_action)*(reward - self.avg_reward[self.last_state])
        self.avg_reward[self.last_state] += self._inverse_count(self.last_state)*(reward - self.avg_reward[self.last_state])
        self.last_state = state
        self.last_action = self.select_action(state)
        return self.last_action
