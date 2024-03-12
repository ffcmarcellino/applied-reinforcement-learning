from envs.minigrid_env import MultiGoalEnv
from agents.td_0 import SarsaAgent, ExpectedSarsaAgent, QLearningAgent, DoubleQLearningAgent
from rl_task import RLTask
from metrics import BaseMetric, NumStepsMetric, ReturnMetric
from visualization import *

NUM_RUNS = 2
NUM_EPISODES = 10
PLOT_PRECISION = 1

class OutcomeMetric(BaseMetric):
    NAME='outcome'
    ONE_HOT_ENCODING = {
        MultiGoalEnv.DOOR_GOAL_REWARD: 1,
        MultiGoalEnv.GOAL_REWARD: 2,
        MultiGoalEnv.LAVA_GOAL_REWARD: 3,
        MultiGoalEnv.LAVA_REWARD: 4,
        MultiGoalEnv.DEFAULT_REWARD: 5
    }
    def update(self, metrics, **kwargs):
        try:
            pos = self.ONE_HOT_ENCODING[kwargs['reward']]
        except:
            pos = 0
        metrics[self.NAME].append(np.eye(len(self.ONE_HOT_ENCODING)+1)[pos])

def moving_average(series):
    moving_avg = [sum(series[:5])/5]
    for i in range(5, len(series)):
        moving_avg.append((moving_avg[-1]*5 - series[i-5] + series[i])/5)
    return moving_avg

def run_experiment(rl_task, metric_objs=None, metric_objs_test=None, num_runs=NUM_RUNS, num_episodes=NUM_EPISODES, random_seed = 0):
    
    np.random.seed(random_seed)
    metrics = {metric.NAME: [] for metric in metric_objs} if metric_objs is not None else {}
    metrics_test = {metric.NAME: [] for metric in metric_objs_test} if metric_objs_test is not None else {}
    episodes_test = []
    
    for i in range(num_runs):
        rl_task.agent.reset()
        for m in metrics:
            metrics[m].append([])
        for m in metrics_test:
            metrics_test[m].append([])
        for j in range(num_episodes+1):
            metrics_j = rl_task.run_episode(metric_objs)
            for m in metrics:
                metrics[m][-1].append(metrics_j[m])
            if j % PLOT_PRECISION == 0:
                metrics_j_test = rl_task.test_episode(metric_objs_test)
                for m in metrics_test:
                    metrics_test[m][-1].append(metrics_j_test[m])
    episodes_test = [x for x in range(1, num_episodes+2, PLOT_PRECISION)]
    for m in metrics:
        metrics[m] = np.array(metrics[m]).mean(axis=0).squeeze()
        metrics[m] = moving_average(metrics[m])
    for m in metrics_test:
        metrics_test[m] = np.array(metrics_test[m]).mean(axis=0).squeeze()
        
    return metrics, metrics_test, episodes_test

def get_agent_info(eps, step_size):
    return {
            'initializer_params': {'init_method': 'zero'},
            'action_selection_params': {
                'method': 'epsilon_greedy',
                'kwargs': {'eps': eps}
            },
            'step_size': step_size
            }

def test_notebook():
    
    env = MultiGoalEnv()
    test_env = MultiGoalEnv(save=True, render_mode='human')
    stochastic_env = MultiGoalEnv(stochastic=True)
    
    num_states = (env.width-2)*(env.height-2)*2 + 2
    num_actions = 5
    metric_objs = [NumStepsMetric('on_episode_end'), ReturnMetric('on_episode_step'), OutcomeMetric('on_episode_end')]
    metric_objs_test = [NumStepsMetric('on_episode_test_end'), ReturnMetric('on_episode_test_step'), OutcomeMetric('on_episode_test_end')]
    eps = 0.1
    step_size = 0.1
    
    sarsa_agent = SarsaAgent(num_states, num_actions, get_agent_info(eps, step_size))
    rl_task_sarsa = RLTask(sarsa_agent, env)
    metric = run_experiment(rl_task_sarsa, metric_objs, metric_objs_test)
    metric[0]['outcome']
    metric[1]['return']
    sarsa_agent.select_action(0, greedy=True)
    env.clean_grids()
    
    expected_sarsa_agent = ExpectedSarsaAgent(num_states, num_actions, get_agent_info(eps, step_size))
    rl_task_expected_sarsa = RLTask(expected_sarsa_agent, env)
    metric = run_experiment(rl_task_expected_sarsa, metric_objs, metric_objs_test)
    metric[0]['return']
    metric[0]['outcome']
    metric[1]['return']
    expected_sarsa_agent.select_action(0, greedy=True)
    
    qlearning_agent = QLearningAgent(num_states, num_actions, get_agent_info(eps, step_size))
    rl_task_qlearning = RLTask(qlearning_agent, env)
    metric = run_experiment(rl_task_qlearning, metric_objs, metric_objs_test)
    metric[0]['return']
    metric[0]['outcome']
    metric[1]['return']
    qlearning_agent.select_action(0, greedy=True)
    
    qlearning_agent_stochastic = QLearningAgent(num_states, num_actions, get_agent_info(0.5, 0.5))
    rl_task_qlearning_stochastic = RLTask(qlearning_agent_stochastic, stochastic_env)
    metric = run_experiment(rl_task_qlearning_stochastic, [OutcomeMetric('on_episode_end')])
    metric[0]['outcome']
    
    double_qlearning_agent = DoubleQLearningAgent(num_states, num_actions, get_agent_info(eps, step_size))
    rl_task_double_qlearning = RLTask(double_qlearning_agent, env)
    metric = run_experiment(rl_task_double_qlearning, metric_objs, metric_objs_test)
    metric[0]['return']
    metric[0]['outcome']
    metric[1]['return']
    
    double_qlearning_agent_stochastic = DoubleQLearningAgent(num_states, num_actions, get_agent_info(0.5, 0.5))
    rl_task_double_qlearning_stochastic = RLTask(double_qlearning_agent_stochastic, stochastic_env)
    metric = run_experiment(rl_task_double_qlearning_stochastic, [OutcomeMetric('on_episode_end')])
    metric[0]['outcome']