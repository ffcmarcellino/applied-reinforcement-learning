import numpy as np
from .base_agent import BaseAgent

class SarsaAgent(BaseAgent):
    
    def step(self, state, reward):
        action = self.select_action(state)
        target_q = reward + self.discount * self.action_values[state, action]
        self.action_values[self.last_state, self.last_action] += self._step_size(self.last_state, self.last_action) * (target_q - self.action_values[self.last_state, self.last_action])
        self.t += 1
        self.num_visits[self.last_state, self.last_action] += 1
        self.last_state = state
        self.last_action = action
        return self.last_action
    
class ExpectedSarsaAgent(BaseAgent):
    
    def __init__(self, num_states, num_actions, agent_info):
        super().__init__(num_states, num_actions, agent_info)
        target_policy_params = agent_info.get('target_policy_params', agent_info['action_selection_params'])
        self.target_policy = self._get_policy_method(target_policy_params['method'], **target_policy_params.get('kwargs', {}))  

    def step(self, state, reward):
        target_policy = self.target_policy(state)
        expected_q = self.action_values[state, target_policy] if isinstance(target_policy, np.int64) else np.sum(target_policy * self.action_values[state])
        target_q = reward + self.discount * expected_q
        self.action_values[self.last_state, self.last_action] += (self._step_size(self.last_state, self.last_action) * (target_q - self.action_values[self.last_state, self.last_action]))
        self.t += 1
        self.num_visits[self.last_state, self.last_action] += 1
        self.last_state = state
        self.last_action = self.select_action(state)
        return self.last_action
    
class QLearningAgent(ExpectedSarsaAgent):
    
    def __init__(self, num_states, num_actions, agent_info):
        agent_info['target_policy_params'] = {'method': 'greedy'}
        super().__init__(num_states, num_actions, agent_info)

class DoubleQLearningAgent(BaseAgent):
    
    def __init__(self, num_states, num_actions, agent_info):
        super().__init__(num_states, num_actions, agent_info)
        self.action_values_1 = None
        self.action_values_2 = None
    
    def reset(self):
        super().reset()
        self.action_values_1 = self.initializer.initialize(**self.initializer_params)
        self.action_values_2 = self.initializer.initialize(**self.initializer_params)
        
    def step(self, state, reward):
        
        if np.random.random() < 0.5:
            a = np.argmax(self.action_values_1[state])
            target_q = reward + self.discount * self.action_values_2[state][a]
            self.action_values_1[self.last_state, self.last_action] += self._step_size(self.last_state, self.last_action) * (target_q - self.action_values_1[self.last_state, self.last_action])
        else:
            a = np.argmax(self.action_values_2[state])
            target_q = reward + self.discount * self.action_values_1[state][a]
            self.action_values_2[self.last_state, self.last_action] += self._step_size(self.last_state, self.last_action) * (target_q - self.action_values_2[self.last_state, self.last_action])
            
        self.action_values[self.last_state, self.last_action] = (self.action_values_1[self.last_state, self.last_action] + self.action_values_2[self.last_state, self.last_action]) / 2
        self.t += 1
        self.num_visits[self.last_state, self.last_action] += 1
        self.last_state = state
        self.last_action = self.select_action(state)
        return self.last_action