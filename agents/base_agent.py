from .action_selection import *
from .initializers import Initializer

class BaseAgent:

    def __init__(self, num_states, num_actions, agent_info):
        self.num_states = num_states
        self.num_actions = num_actions
        self.initializer = Initializer(self.num_states, self.num_actions)
        self.initializer_params = agent_info['initializer_params']
        self.get_policy = self._get_policy_method(agent_info['action_selection_params']['method'], **agent_info['action_selection_params'].get('kwargs', {}))
        self._step_size = self._get_step_size_fun(agent_info.get('step_size', '1/n'))
        self.discount = agent_info.get('discount', 1)
        self.allowed_actions_mask = agent_info['action_selection_params'].get('allowed_actions_mask', None)

        self.t = None
        self.num_visits = None
        self.action_values = None
        self.last_state = None
        self.last_action = None

    def reset(self):
        self.t = 0
        self.num_visits = self.initializer.initialize("zero")
        self.action_values = self.initializer.initialize(**self.initializer_params)
    
    def start(self, init_state):
        self.last_state = init_state
        self.last_action = self.select_action(init_state)
        return self.last_action

    def step(self, state, reward):
        raise NotImplementedError()

    def on_episode_end(self):
        return

    def select_action(self, state, greedy=False):
        policy = self.get_policy(state) if not greedy else self.get_greedy_policy(state, tie_break='random')
        if isinstance(policy, np.int64):
            return policy
        else:
            return policy_action_selection(policy)

    def get_greedy_policy(self, states, **greedy_policy_kwargs):
        mask = None if self.allowed_actions_mask is None else self.allowed_actions_mask[states]
        return greedy_policy(self.action_values[states], mask=mask, **greedy_policy_kwargs)

    def get_epsilon_greedy_policy(self, states, **epsilon_greedy_policy_kwargs):
        mask = None if self.allowed_actions_mask is None else self.allowed_actions_mask[states]
        eps_fun = epsilon_greedy_policy_kwargs.get('fun', None)
        eps = epsilon_greedy_policy_kwargs['eps']
        if eps_fun is not None:
            eps /= eps_fun(self.t)
        policy_kwargs = {k: v for (k,v) in epsilon_greedy_policy_kwargs.items() if k not in ('eps', 'fun')}
        return epsilon_greedy_policy(self.action_values[states], eps, mask=mask, **policy_kwargs)

    def get_ucb_policy(self, states, **ucb_policy_kwargs):
        mask = None if self.allowed_actions_mask is None else self.allowed_actions_mask[states]
        return ucb_policy(self.action_values[states], self.num_visits[states], mask=mask,  **ucb_policy_kwargs)

    def get_softmax_policy(self, states):
        mask = None if self.allowed_actions_mask is None else self.allowed_actions_mask[states]
        return softmax_policy(self.action_values[states], mask=mask)

    def _inverse_count(self, state=None, action=None):

        if state is None:
            return 1/self.t
        else:
            return 1/self.num_visits[state].sum() if action is None else 1/self.num_visits[state][action]

    def _get_step_size_fun(self, step_size_f):

        if type(step_size_f) in (int, float):
            def step_size_fun(s,a):
                return step_size_f

        elif step_size_f == '1/n':
            return self._inverse_count

        else:
            def step_size_fun(s,a):
                return step_size_f(self._inverse_count(s,a))

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
