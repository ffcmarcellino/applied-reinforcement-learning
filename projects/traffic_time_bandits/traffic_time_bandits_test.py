import pandas as pd
import numpy as np

from envs.traffic_time_env import TrafficTimeEnv, ContextualTrafficTimeEnv
from agents.bandits import KBandits, GradientBandits
from rl_task import RLTask
from visualization import *

def test_traffic_time_bandits_notebook():

    env = TrafficTimeEnv()

    num_actions = 5
    num_runs = 2
    num_steps = 2

    metrics_info = {
        'optimal_action_flag': {'optimal_action': 2},
        'final_policy': {'states': 0},
        'cum_avg_reward': None
    }

    env.AVGS_RAIN
    env.P_RAIN
    env.AVGS_NO_RAIN
    env.P_RAIN

    optimal_avg_reward = {
    'x': num_steps,
    'y': np.ones(num_steps),
    }

    name = ""
    title = ""

    greedy_agent_info = {
    'initializer_params': {'init_method': 'constant', 'constant': -26},
    'action_selection_params': {
        'method': 'greedy',
        'kwargs': {'tie_break': 'random'}
    },
    'step_size': '1/n'
    }

    greedy_agent = KBandits(1, num_actions, greedy_agent_info)
    task = RLTask(greedy_agent, env)
    metrics = task.run_independent_mdps(num_runs, num_steps, metrics_info)

    metrics['final_policy']

    cum_avg_reward_1 ={
    'x': num_steps,
    'y': metrics['cum_avg_reward'],
    'name': name,
    'color': 0
    }

    plot_cum_avg_reward([optimal_avg_reward, cum_avg_reward_1], title)

    perc_opt_actions_1 ={
    'x': num_steps,
    'y': metrics['optimal_action_flag'],
    'name': name,
    'color': 11
    }

    plot_perc_opt_actions(data=[perc_opt_actions_1], title=title)

    epsilon_greedy_agent_info = {
    'initializer_params': {'init_method': 'constant', 'constant': -26},
    'action_selection_params': {
        'method': 'epsilon_greedy',
        'kwargs': {'eps': 0.01}
    },
    'step_size': '1/n'
    }

    epsilon_greedy_agent = KBandits(1, num_actions, epsilon_greedy_agent_info)
    task = RLTask(epsilon_greedy_agent, env)
    metrics = task.run_independent_mdps(num_runs, num_steps, metrics_info)

    ucb_agent_info = {
    'initializer_params': {'init_method': 'constant', 'constant': -26},
    'action_selection_params': {
        'method': 'ucb',
        'kwargs': {'c': 1, 'tie_break': 'random'}
    },
    'step_size': '1/n'
    }

    ucb_agent = KBandits(1, num_actions, ucb_agent_info)
    task = RLTask(ucb_agent, env)
    metrics = task.run_independent_mdps(num_runs, num_steps, metrics_info)

    gradient_agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'step_size': 0.005
    }

    gradient_agent = GradientBandits(1, num_actions, gradient_agent_info)
    task = RLTask(gradient_agent, env)
    metrics = task.run_independent_mdps(num_runs, num_steps, metrics_info)

    env = ContextualTrafficTimeEnv()

    metrics_info = {
    'cum_avg_reward': None,
    'final_policy': {'states':[0,1]}
    }

    ucb_agent_info = {
    'initializer_params': {'init_method': 'constant', 'constant': -26},
    'action_selection_params': {
        'method': 'ucb',
        'kwargs': {'c': 10, 'tie_break': 'random'}
    },
    'step_size': '1/n'
    }

    ucb_agent = KBandits(2, num_actions, ucb_agent_info)
    task = RLTask(ucb_agent, env)
    metrics = task.run_independent_mdps(num_runs, num_steps, metrics_info)
