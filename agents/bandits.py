from .base_agent import BaseAgent

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
        self.avg_reward = self.initializer.zero_initializer(self.num_states)
        return super().reset(init_state)

    def step(self, state, reward):
        self.t += 1
        self.num_visits[self.last_state, self.last_action] += 1
        self.action_values[self.last_state] -= self._step_size(self.last_state, self.last_action)*(reward - self.avg_reward[self.last_state])*self.get_policy(self.last_state)
        self.action_values[self.last_state, self.last_action] += self._step_size(self.last_state, self.last_action)*(reward - self.avg_reward[self.last_state])
        self.avg_reward[self.last_state] += self._inverse_count(self.last_state)*(reward - self.avg_reward[self.last_state])
        self.last_state = state
        self.last_action = self.select_action(state)
        return self.last_action
