import numpy as np
from .base_agent import BaseAgent

class MonteCarloAgent(BaseAgent):

    def __init__(self, num_states, num_actions, agent_info):
        super().__init__(num_states, num_actions, agent_info)
        self.exploring_starts = agent_info.get('exploring_starts', False)
        self.state_action_to_after_state_map = agent_info.get('state_action_to_after_state_map', None)
        self.trajectory = None

    def reset(self, init_state, hard_reset=False):
        if hard_reset or self.num_visits is None or self.action_values is None:
            self.t = 1
            self.num_visits = self.initializer.initialize("zero")
            self.action_values = self.initializer.initialize(**self.initializer_params)
        self.last_state = init_state
        if self.exploring_starts:
            self.last_action = np.random.randint(self.num_actions) if self.allowed_actions_mask is None else np.random.choice(np.nonzero(self.allowed_actions_mask[init_state])[0])
        else:
            self.last_action = self.select_action(init_state)
        self.trajectory = [(None, init_state, self.last_action)]
        return self.last_action

    def step(self, state, reward):
        self.last_state = state
        self.last_action = self.select_action(state)
        self.trajectory.append((reward, state, self.last_action))
        return self.last_action

    def on_episode_end(self):
        if self.state_action_to_after_state_map is not None:
            self._learn_from_trajectory_after_state()
        else:
            self._learn_from_trajectory()

    def _learn_from_trajectory(self):

        first_visits = {}
        for i, (_,s,a) in enumerate(self.trajectory):
            if s not in first_visits:
                first_visits[s] = {}
            if a not in first_visits[s]:
                first_visits[s][a] = i

        G = 0
        for t in range(len(self.trajectory)-2, -1, -1):
            r = self.trajectory[t+1][0]
            s, a = self.trajectory[t][1:]
            G = G*self.discount + r
            if t == first_visits[s][a]:
                self.num_visits[s][a] += 1
                self.action_values[s][a] += self._step_size(s, a)*(G - self.action_values[s][a])
        self.t += 1

    def _learn_from_trajectory_after_state(self):

        first_visits = {}
        for i, (_,s,a) in enumerate(self.trajectory):
            after_state = self.state_action_to_after_state_map[s,a]
            if after_state not in first_visits:
                first_visits[after_state] = i

        G = 0
        for t in range(len(self.trajectory)-2, -1, -1):
            r = self.trajectory[t+1][0]
            s, a = self.trajectory[t][1:]
            after_state = self.state_action_to_after_state_map[s,a]
            G = G*self.discount + r
            if t == first_visits[after_state]:
                self.num_visits[s][a] += 1
                self.action_values[s][a] += self._step_size(s, a)*(G - self.action_values[s][a])
                all_s, all_a = np.nonzero(self.state_action_to_after_state_map == after_state)
                self.num_visits[all_s, all_a] = self.num_visits[s][a]
                self.action_values[all_s, all_a] = self.action_values[s][a]
        self.t += 1