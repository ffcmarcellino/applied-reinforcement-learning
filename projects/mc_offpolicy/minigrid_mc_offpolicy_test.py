import numpy as np
from envs.minigrid_env import LavaEnv
from agents.monte_carlo import MonteCarloAgent
from rl_task import RLTask
from metrics import ReturnMetric, NumStepsMetric, NumUpdatesMetric, ActionValuesMetric

NUM_EPISODES_TRAIN = 10
NUM_EPISODES_TEST = 1
PLOT_PRECISION = 2
MAX_NUM_VALUES = 5

env = LavaEnv(max_steps=100)

init_state,_ = env.reset()
num_states = (env.height-2)*(env.width-2)
num_actions = 4
metric_objs_test = [ReturnMetric('on_episode_test_step'), NumStepsMetric('on_episode_test_end')]
metric_objs_offpolicy = [ActionValuesMetric('on_episode_end'), NumUpdatesMetric('on_episode_end')]
PAIRS = {(init_state, 3)}

def update_action_values(metrics, metrics_i, pairs=None):
    
    states, actions = np.nonzero(metrics_i['num_updates'] > 0)
    
    for s,a in zip(states, actions):
        if pairs is None or (pairs is not None and (s,a) in pairs):
            if (s,a) not in metrics['action_values']:
                metrics['action_values'][(s,a)] = [[metrics_i['action_values'][s,a]], metrics_i['num_updates'][s,a]]
            elif (metrics_i['num_updates'][s,a] > metrics['action_values'][(s,a)][-1]) and (len(metrics['action_values'][(s,a)][0]) < MAX_NUM_VALUES):
                metrics['action_values'][(s,a)][0].append(metrics_i['action_values'][s,a])
                metrics['action_values'][(s,a)][-1] = metrics_i['num_updates'][s,a]
                
def run_experiment(rl_task, metric_objs=None, metric_objs_test=None, random_seed=0):
    
    np.random.seed(random_seed)
    episodes = []
    metrics = {metric.NAME: [] for metric in metric_objs} if metric_objs is not None else {}
    metrics_test = {metric.NAME: [] for metric in metric_objs_test} if metric_objs_test is not None else {}
    if 'action_values' in metrics: metrics['action_values'] = {}
    max_deltas = []
    
    rl_task.agent.reset(init_state, hard_reset=True)
    
    metrics_i = {metric.NAME: [] for metric in metric_objs_test}
    for i in range(NUM_EPISODES_TEST):
        metrics_i_test = rl_task.test_episode(metric_objs_test)
        for metric in metric_objs_test:
            try:
                metrics_i[metric.NAME] += metrics_i_test[metric.NAME]
            except:
                metrics_i[metric.NAME].append(metrics_i_test[metric.NAME])
            
    episodes.append(0)          
    
    for metric in metric_objs_test:
        metrics_test[metric.NAME].append(np.mean(metrics_i[metric.NAME]))
        
    if 'num_updates' in metrics: metrics['num_updates'].append(0)
    
    for i in range(PLOT_PRECISION):
        
        action_values_prev = rl_task.agent.action_values.copy()
        
        num_updates = None
        for j in range(NUM_EPISODES_TRAIN // PLOT_PRECISION):
            metrics_j = rl_task.run_episode(metric_objs)
            if (j == 0) and ('num_updates' in metrics): num_updates = metrics_j['num_updates'].sum()
            if 'action_values' in metrics: update_action_values(metrics, metrics_j, PAIRS)
        if 'num_updates' in metrics:
            metrics['num_updates'][-1] = num_updates - metrics['num_updates'][-1]
            metrics['num_updates'].append(metrics_j['num_updates'].sum())
        
        max_delta = np.abs(action_values_prev-rl_task.agent.action_values).max()
        
        metrics_i = {metric.NAME: [] for metric in metric_objs_test}
        for k in range(NUM_EPISODES_TEST):
            metrics_i_test = rl_task.test_episode(metric_objs_test)
            for metric in metric_objs_test:
                try:
                    metrics_i[metric.NAME] += metrics_i_test[metric.NAME]
                except:
                    metrics_i[metric.NAME].append(metrics_i_test[metric.NAME])
            
        episodes.append((i+1)*(NUM_EPISODES_TRAIN // PLOT_PRECISION))
        max_deltas.append(max_delta)
        
        for metric in metric_objs_test:
            metrics_test[metric.NAME].append(np.mean(metrics_i[metric.NAME]))
        
    if 'num_updates' in metrics:
        metrics_j = rl_task.run_episode(metric_objs)
        
        metrics['num_updates'][-1] = metrics_j['num_updates'].sum() - metrics['num_updates'][-1]
        
    return episodes, metrics, metrics_test

def linear_decay(eps, start_x, start_y, delta_x, delta_y):
    return lambda x: eps / (delta_y * (x - start_x) / delta_x + start_y)

def test_notebook():

    env.pprint_grid().split("\n")

    mc_onpolicy_info = {
        'initializer_params': {'init_method': 'zero'},
        'action_selection_params': {
            'method': 'epsilon_greedy',
            'kwargs': {'eps': 0.3}
        },
        'step_size': '1/n'
    }

    mc_onpolicy_agent = MonteCarloAgent(num_states, num_actions, mc_onpolicy_info)

    rl_task_onpolicy = RLTask(mc_onpolicy_agent, env)

    episodes_onpolicy, _, metrics_test_onpolicy = run_experiment(rl_task_onpolicy, None, metric_objs_test, random_seed=100)

    (rl_task_onpolicy.agent.action_values == 0).mean()

    rl_task_onpolicy.env.clean_grids()

    eps = 0.5

    mc_onpolicy_decay_info = {
        'initializer_params': {'init_method': 'zero'},
        'action_selection_params': {
            'method': 'epsilon_greedy',
            'kwargs': {'eps': eps, 'fun': linear_decay(eps, 0, eps, NUM_EPISODES_TRAIN, -eps + 0.000001)}
        },
        'step_size': '1/n'
    }

    mc_onpolicy_decay_agent = MonteCarloAgent(num_states, num_actions, mc_onpolicy_decay_info)

    rl_task_onpolicy_decay = RLTask(mc_onpolicy_decay_agent, env)

    episodes_onpolicy_decay, _, metrics_test_onpolicy_decay = run_experiment(rl_task_onpolicy_decay, None, metric_objs_test, random_seed=100)
    
    mc_offpolicy_ois_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {
        'method': 'epsilon_greedy',
        'kwargs': {'eps': 0.3}
    },
    'learning_params': {'method': 'off_policy', 'type': 'importance_sampling', 'average_type': 'ordinary'},
    'step_size': '1/n'
}

    mc_offpolicy_ois_agent = MonteCarloAgent(num_states, num_actions, mc_offpolicy_ois_info)

    rl_task_offpolicy_ois = RLTask(mc_offpolicy_ois_agent, env)

    episodes_offpolicy_ois, metrics_offpolicy_ois, metrics_offpolicy_ois_test = run_experiment(rl_task_offpolicy_ois, metric_objs_offpolicy, metric_objs_test, random_seed=100)
    
    mc_offpolicy_daois_info = {
        'initializer_params': {'init_method': 'zero'},
        'action_selection_params': {
            'method': 'epsilon_greedy',
            'kwargs': {'eps': 0.3}
        },
        'learning_params': {'method': 'off_policy', 'type': 'discounting_aware_importance_sampling', 'average_type': 'ordinary'},
        'step_size': '1/n',
        'discount': 0.99
    }

    mc_offpolicy_daois_agent = MonteCarloAgent(num_states, num_actions, mc_offpolicy_daois_info)

    rl_task_offpolicy_daois = RLTask(mc_offpolicy_daois_agent, env)
    
    episodes_offpolicy_daois, metrics_offpolicy_daois, metrics_offpolicy_daois_test = run_experiment(rl_task_offpolicy_daois, metric_objs_offpolicy, metric_objs_test, random_seed=100)

    mc_offpolicy_dawis_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {
        'method': 'epsilon_greedy',
        'kwargs': {'eps': 0.3}
    },
    'learning_params': {'method': 'off_policy', 'type': 'discounting_aware_importance_sampling', 'average_type': 'weighted'},
    'step_size': '1/n',
    'discount': 0.99
}

    mc_offpolicy_dawis_agent = MonteCarloAgent(num_states, num_actions, mc_offpolicy_dawis_info)

    rl_task_offpolicy_dawis = RLTask(mc_offpolicy_dawis_agent, env)
    
    episodes_offpolicy_dawis, metrics_offpolicy_dawis, metrics_offpolicy_dawis_test = run_experiment(rl_task_offpolicy_dawis, metric_objs_offpolicy, metric_objs_test, random_seed=100)
    
    mc_offpolicy_pdis_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {
        'method': 'epsilon_greedy',
        'kwargs': {'eps': 0.3}
    },
    'learning_params': {'method': 'off_policy', 'type': 'per_decision_importance_sampling'},
    'step_size': '1/n'
}

    mc_offpolicy_pdis_agent = MonteCarloAgent(num_states, num_actions, mc_offpolicy_pdis_info)

    rl_task_offpolicy_pdis = RLTask(mc_offpolicy_pdis_agent, env)
    
    episodes_offpolicy_pdis, metrics_offpolicy_pdis, metrics_offpolicy_pdis_test = run_experiment(rl_task_offpolicy_pdis, metric_objs_offpolicy, metric_objs_test, random_seed=100)
    
