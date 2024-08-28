import numpy as np
from collections import deque
from .base_agent import BaseAgent

class nStepBaseAgent(BaseAgent):

    def __init__(self, num_states, num_actions, agent_info):
        super().__init__(num_states, num_actions, agent_info)
        self.num_steps = agent_info.get("num_steps", None)
        self.get_sigma = agent_info.get("sigma_func", lambda i: 1)

    def reset(self):
        super().reset()
        self.trajectory = deque()

    def start(self, init_state):
        self.trajectory = deque()
        self.t = 0
        return super().start(init_state)

    def step(self, state, reward):
        self.t += 1
        action = self.select_action(state)
        self.trajectory.append((reward, state, action))
        if self.num_steps is not None:
            if self.t >= self.num_steps:
                if self.t > self.num_steps:
                    _, self.last_state, self.last_action = self.trajectory.popleft()
                target_q = self._learn_from_trajectory(update_all=False)
                self.update_action_value(self.last_state, self.last_action, target_q)
        return action

    def on_episode_end(self):
        self._learn_from_trajectory(update_all=True)
        self.episode_num += 1

    def _get_importance_sampling_ratio(self, s, a):
        target_prob = self.get_action_prob(s, a, "target")
        behavior_prob = self.get_action_prob(s, a, "behavior")
        return target_prob/behavior_prob

    def _learn_from_trajectory(self, update_all):
        raise NotImplementedError()

    def _nstep_onpolicy_recursion(self, i, update):
        if i == len(self.trajectory):
            _,s,a = self.trajectory[i-1]
            return self.action_values[s,a]
        r,s,a = self.trajectory[i]
        target_q = self._nstep_onpolicy_recursion(i+1, update)
        if update:
            self.update_action_value(s, a, target_q)
        return r + self.discount * target_q

    def _nstep_offpolicy_recursion(self, i, update):
        sigma = self.get_sigma(i)
        if i == len(self.trajectory):
            _,s,a = self.trajectory[i-1]
            return self.action_values[s,a]
        r,s,a = self.trajectory[i]
        isr = self._get_importance_sampling_ratio(s,a)
        k = isr if sigma == 1 else self.get_action_prob(s,a, "target")
        if update:
            target_q = r + self.discount*k*(self._nstep_offpolicy_recursion(i+1, update) - self.action_values[s,a]) + self.discount*self.get_value(s)
            self.update_action_value(s, a, target_q)
        if k == 0:
           return r + self.discount*self.get_value(s)
        if not update:
           target_q = r + self.discount*k*(self._nstep_offpolicy_recursion(i+1, update) - self.action_values[s,a]) + self.discount*self.get_value(s)
        return target_q

class nStepSarsaAgent(nStepBaseAgent):

    def __init__(self, num_states, num_actions, agent_info):
        super().__init__(num_states, num_actions, agent_info)
        self._learn_from_trajectory = lambda update_all: self._nstep_onpolicy_recursion(0, update_all)

class nStepQLearningAgent(nStepBaseAgent):

    def __init__(self, num_states, num_actions, agent_info):
        agent_info['target_policy_params'] = {'method': 'greedy'}
        super().__init__(num_states, num_actions, agent_info)
        self._learn_from_trajectory = lambda update_all: self._nstep_offpolicy_recursion(0, update_all)
