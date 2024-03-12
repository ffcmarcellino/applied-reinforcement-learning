import numpy as np
from .base_agent import BaseAgent

class MonteCarloAgent(BaseAgent):

    def __init__(self, num_states, num_actions, agent_info):
        super().__init__(num_states, num_actions, agent_info)
        self.exploring_starts = agent_info.get('exploring_starts', False)
        self.learning_params = agent_info.get('learning_params', {'method': 'on_policy'})
        self.state_action_to_after_state_map = agent_info.get('state_action_to_after_state_map', None)
        self.trajectory = None
        self._learn_from_trajectory = self._get_learning_fun()

    def reset(self):
        self.t = 1
        self.num_visits = self.initializer.initialize("zero")
        if self.learning_params.get('average_type', None) == 'weighted':
            self.weight_sum = self.initializer.initialize("zero")
        self.action_values = self.initializer.initialize(**self.initializer_params)
        
    def start(self, init_state):
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
        self._learn_from_trajectory()

    def _get_learning_fun(self):
        
        if self.learning_params['method'] == 'on_policy':
            if self.state_action_to_after_state_map is not None:
                return self._on_policy_learning_after_state
            return self._on_policy_learning

        elif self.learning_params['method'] == 'off_policy':

            if self.learning_params['type'] == 'importance_sampling':
                if self.learning_params['average_type'] == 'ordinary':
                    return self._ordinary_importance_sampling
                elif self.learning_params['average_type'] == 'weighted':
                    return self._weighted_importance_sampling
                else:
                    raise(f"Average type '{self.learning_params['average_type']}' is not supported.")

            elif self.learning_params['type'] == 'discounting_aware_importance_sampling':
                if self.learning_params['average_type'] == 'ordinary':
                    return self._discounting_aware_ordinary_importance_sampling
                elif self.learning_params['average_type'] == 'weighted':
                    return self._discounting_aware_weighted_importance_sampling
                else:
                    raise(f"Average type '{self.learning_params['average_type']}' is not supported.")
            
            elif self.learning_params['type'] == 'per_decision_importance_sampling':
                return self._per_decision_importance_sampling

            else:
                raise(f"Type '{self.learning_params['type']}' is not supported.")

        else:
            raise(f"Method '{self.learning_params['method']}' is not supported.")
    

    def _on_policy_learning(self):

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

    def _on_policy_learning_after_state(self):

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

    def _ordinary_importance_sampling(self):

        G = 0
        W = 1
        for t in range(len(self.trajectory)-2, -1, -1):
            r = self.trajectory[t+1][0]
            s, a = self.trajectory[t][1:]
            G = G*self.discount + r 
            self.num_visits[s][a] += 1
            self.action_values[s][a] += self._inverse_count(s,a)*(W*G - self.action_values[s][a])
            if self.get_greedy_policy(s) != a:
                break
            W /= self.get_policy(s)[a]
        self.t += 1

    def _weighted_importance_sampling(self):

        G = 0
        W = 1
        for t in range(len(self.trajectory)-2, -1, -1):
            r = self.trajectory[t+1][0]
            s, a = self.trajectory[t][1:]
            G = G*self.discount + r
            self.weight_sum[s][a] += W   
            self.num_visits[s][a] += 1
            self.action_values[s][a] += W*(G - self.action_values[s][a])/self.weight_sum[s][a]
            if self.get_greedy_policy(s) != a:
                break
            W /= self.get_policy(s)[a]
        self.t += 1

    def _discounting_aware_ordinary_importance_sampling(self):

        cum_rewards = np.array([])
        term_degrees = np.array([1])
        cum_importances = np.array([1])
        for t in range(len(self.trajectory)-2, -1, -1):
            r = self.trajectory[t+1][0]
            s, a = self.trajectory[t][1:]
            cum_rewards = np.r_[cum_rewards, 0]
            cum_rewards += r
            weighted_G = np.sum(cum_rewards * term_degrees * cum_importances)
            self.num_visits[s][a] += 1
            self.action_values[s][a] += self._inverse_count(s,a)*(weighted_G - self.action_values[s][a])
            term_degrees = np.r_[self.discount * term_degrees, 1-self.discount]
            cum_importances = np.r_[cum_importances * (self.get_greedy_policy(s) == a) / self.get_policy(s)[a], 1]
        self.t += 1

    def _discounting_aware_weighted_importance_sampling(self):

        cum_rewards = np.array([])
        term_degrees = np.array([1])
        cum_importances = np.array([1])
        for t in range(len(self.trajectory)-2, -1, -1):
            r = self.trajectory[t+1][0]
            s, a = self.trajectory[t][1:]
            cum_rewards = np.r_[cum_rewards, 0]
            cum_rewards += r
            weights = term_degrees * cum_importances
            weighted_G = np.sum(cum_rewards * weights)
            self.weight_sum[s][a] += weights.sum()
            self.num_visits[s][a] += 1
            self.action_values[s][a] += weights.sum()*(weighted_G/weights.sum() - self.action_values[s][a]) / self.weight_sum[s][a]
            term_degrees = np.r_[self.discount * term_degrees, 1-self.discount]
            cum_importances = np.r_[cum_importances * (self.get_greedy_policy(s) == a) / self.get_policy(s)[a], 1]
        self.t += 1

    def _per_decision_importance_sampling(self):

        weighted_G = 0
        W = 1
        for t in range(len(self.trajectory)-2, -1, -1):
            r = self.trajectory[t+1][0]
            s, a = self.trajectory[t][1:]
            weighted_G = weighted_G*self.discount*W + r 
            self.num_visits[s][a] += 1
            self.action_values[s][a] += self._inverse_count(s,a)*(weighted_G - self.action_values[s][a])
            W = (self.get_greedy_policy(s) == a)/self.get_policy(s)[a]
        self.t += 1

