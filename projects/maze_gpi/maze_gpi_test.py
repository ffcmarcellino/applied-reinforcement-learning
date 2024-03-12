import numpy as np
import time
import gym
import gym_maze
from tqdm import tqdm

from rl_task import RLTask
from agents.generalized_policy_iteration import PolicyIteration, ValueIteration
from agents.action_selection import policy_action_selection
from metrics import NumStepsMetric
from visualization import plot_figure

class GPIAgent:

    def __init__(self, policy, obs_to_state_fun):
        self.policy = policy
        self.obs_to_state = obs_to_state_fun
        
    def reset(self):
        self.t = 0

    def start(self, init_obs):
        action = policy_action_selection(self.policy[self.obs_to_state(init_obs)])
        return int(action)

    def step(self, next_obs, reward):
        action = policy_action_selection(self.policy[self.obs_to_state(next_obs)])
        self.t += 1
        return int(action)

    def on_episode_end(self):
        return

env = gym.make("maze-sample-10x10-v0", enable_render=False)
num_states = np.product(env.maze_size)
num_actions = len(env.ACTION)
initializer_params = {'init_method': 'zero'}
terminal_states = [env.obs_to_state(env.maze_view.goal)]
theta = 10e-4
next_state_dynamics, reward_dynamics = env.get_dynamics()
metrics_info = {'num_iterations': None,
                'rmse': {'target_state_values': np.zeros(num_states)},
                'max_abs_err': {'target_state_values': np.zeros(num_states)}
                }
metric_objs = [NumStepsMetric('on_episode_end')]
title = ""

def test_plots():

    for color in ['blue', 'darkblue', 'red', 'purple', 'green', 'lightgreen', 'salmon', 'darksalmon', 'goldenrod', 'darkgoldenrod', 'violet', 'brown', 'gray']:
        data = [{
            'x': list(range(10)),
            'y': list(range(10)),
            'color': color
            }]
        plot_figure(data, title=title, xtitle=title, ytitle=title, test=True)

    data = [{
        'x': list(range(10)),
        'y': list(range(10)),
        'name': '',
        'color': 'blue'
        },
        {
        'x': list(range(10)),
        'y': list(range(10)),
        'name': '',
        'color': 'darkblue'
        }]
    plot_figure(data, title=title, xtitle=title, ytitle=title, test=True)

def test_run_polit():

    polit_10 = PolicyIteration(num_states, num_actions, initializer_params, terminal_states, theta, reward_dynamics, next_state_dynamics)
    polit_10.reset()
    polit_10_metrics = polit_10.run(metrics_info = metrics_info)

    polit_10.state_values
    polit_10.policy
    polit_10_metrics['num_iterations'][0]
    polit_10_metrics['rmse']
    polit_10_metrics['max_abs_err']

    polit_10.policy_evaluation()
    polit_10.policy_improvement()

    polit_10_agent = GPIAgent(polit_10.policy, env.obs_to_state)
    rl_task = RLTask(polit_10_agent, env, num_states)
    rl_task.agent.reset()
    metrics = rl_task.run_episode(metric_objs=metric_objs)
    metrics['num_steps']

    polit_10.reset()
    polit_10_metrics = polit_10.run(max_eval_iterations=5, metrics_info = metrics_info)

    polit_10.state_values
    polit_10.policy
    polit_10_metrics['num_iterations'][0]
    polit_10_metrics['rmse']
    polit_10_metrics['max_abs_err']

def test_run_valit():

    valit_10 = ValueIteration(num_states, num_actions, initializer_params, terminal_states, theta, reward_dynamics, next_state_dynamics)
    valit_10.reset()
    valit_10_metrics = valit_10.run(metrics_info=metrics_info)

    valit_10_metrics['rmse']
    valit_10_metrics['max_abs_err']

    valit_10_agent = GPIAgent(valit_10.policy, env.obs_to_state)
    rl_task = RLTask(valit_10_agent, env, num_states)
    rl_task.agent.reset()
    metrics = rl_task.run_episode(metric_objs=metric_objs)
    metrics['num_steps']
