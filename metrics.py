from token import NAME
import numpy as np

class BaseMetric:
    
    def __init__(self, location):
        self.location = location
        
    def update(self, metric, **kwargs):
        raise NotImplementedError
    
class NumStepsMetric(BaseMetric):
    """Number of steps that the agent performed so far."""
    
    NAME = 'num_steps'

    def update(self, metrics, **kwargs):
        metrics[self.NAME] = kwargs['step']
        
class PercPairVisitsMetric(BaseMetric):
    """Percentage of state-action pairs visited by the agent at least once."""
    
    NAME = 'perc_pair_visits'

    def update(self, metrics, **kwargs):
        num_pairs = np.prod(kwargs['agent'].action_values.shape) if kwargs['agent'].allowed_actions_mask is None else kwargs['agent'].allowed_actions_mask.sum()
        metrics[self.NAME] = (kwargs['agent'].num_visits != 0).sum() / num_pairs

class OptimalActionFlagMetric(BaseMetric):
    """Whether the current action is optimal or not."""
        
    NAME = 'optimal_action_flag'
    
    def __init__(self, optimal_action, location):
        super().__init__(location)
        self.optimal_action = optimal_action
    
    def update(self, metrics, **kwargs):
        metrics[self.NAME].append((kwargs['action']==self.optimal_action)*100)
        
class CumAvgRewardMetric(BaseMetric):
    """Cumulative average reward"""
    
    NAME = 'cum_avg_reward'
    
    def update(self, metrics, **kwargs):
        avg_reward = 0 if len(metrics[self.NAME]) == 0 else metrics[self.NAME][-1]
        avg_reward += (kwargs['reward'] - avg_reward)/kwargs['step']
        metrics[self.NAME].append(avg_reward)
        
class GreedyPolicyMetric(BaseMetric):
    """Greedy policy wrt. the action values of the agent."""
    
    NAME = 'greedy_policy'
    
    def __init__(self, states, location):
        super().__init__(location)
        self.states = states
        
    def update(self, metrics, **kwargs):
        metrics[self.NAME] = kwargs['agent'].get_policy(self.states)
        
class ReturnMetric(BaseMetric):
    """Return of the agent"""
    
    NAME = 'return'
    
    def update(self, metrics, **kwargs):
        metrics[self.NAME].append(kwargs['reward'])
        metrics[self.NAME] = [sum(metrics[self.NAME])]
        
class RewardMetric(BaseMetric):
    """Rewards obtained by the agent."""
    
    NAME = 'reward'
    
    def update(self, metrics, **kwargs):
        metrics[self.NAME].append(kwargs['reward'])
        
class ActionValuesMetric(BaseMetric):
    """Action-values computed by the agent"""
    
    NAME = 'action_values'
    
    def update(self, metrics, **kwargs):       
        metrics[self.NAME] = kwargs['agent'].action_values
            
class NumUpdatesMetric(BaseMetric):
     """Number of updates to state-action pairs."""
     
     NAME = 'num_updates'

     def update(self, metrics, **kwargs):   
        metrics[self.NAME] = kwargs['agent'].num_visits
        
class StatesMetric(BaseMetric):
    """Past states visited by the agent."""
    
    NAME = 'states'
    
    def update(self, metrics, **kwargs):
        metrics[self.NAME].append(kwargs['state'])
        
class InfosMetric(BaseMetric):
    """Past info obtained from agent."""
    
    NAME = 'infos'
    
    def update(self, metrics, **kwargs):
        metrics[self.NAME].append(kwargs['info'])