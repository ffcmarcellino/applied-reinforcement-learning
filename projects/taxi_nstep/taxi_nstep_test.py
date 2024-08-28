from agents.nstep_td import nStepSarsaAgent, nStepQLearningAgent
from envs.taxi_env import TaxiEnv
from rl_task import RLTask
from metrics import NumStepsMetric, ReturnMetric
from visualization import *

NUM_RUNS = 2
NUM_EPISODES = 6
PLOT_PRECISION = 1
MAX_EPISODE_STEPS = 10

env = TaxiEnv()
test_env = TaxiEnv(render_mode='human')

num_states = env.observation_space.n
num_actions = env.action_space.n
metric_objs = [NumStepsMetric('on_episode_end'), ReturnMetric('on_episode_step')]
metric_objs_test = [NumStepsMetric('on_episode_test_end'), ReturnMetric('on_episode_test_step')]
episodes = [x for x in range(PLOT_PRECISION, NUM_EPISODES+1)]
eps = 0.3
step_size = 0.5

def moving_average(series):
    moving_avg = [sum(series[:2])/2]
    for i in range(2, len(series)):
        moving_avg.append((moving_avg[-1]*2 - series[i-2] + series[i])/2)
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
                rl_task.env.set_seed(random_seed)
                metrics_j_test = rl_task.test_episode(metric_objs_test)
                rl_task.env.set_seed(None)
                for m in metrics_test:
                    metrics_test[m][-1].append(metrics_j_test[m])
    episodes_test = [x for x in range(1, num_episodes+2, PLOT_PRECISION)]
    for m in metrics:
        metrics[m] = np.array(metrics[m]).mean(axis=0).squeeze()
        metrics[m] = moving_average(metrics[m])
    for m in metrics_test:
        metrics_test[m] = np.array(metrics_test[m]).mean(axis=0).squeeze()
        
    return metrics, metrics_test, episodes_test

def get_td0_agent_info(eps, step_size):
    return  {
            'initializer_params': {'init_method': 'zero'},
            'action_selection_params': {
                'method': 'epsilon_greedy',
                'kwargs': {'eps': eps}
            },
            'step_size': step_size,
            'num_steps': 1
            }


sarsa_agent = nStepSarsaAgent(num_states, num_actions, get_td0_agent_info(eps, step_size))
rl_task_sarsa = RLTask(sarsa_agent, env, MAX_EPISODE_STEPS)
metrics_sarsa = run_experiment(rl_task_sarsa, metric_objs, metric_objs_test, random_seed=20)
metrics_sarsa[0]['return']
metrics_sarsa[1]['return']
metrics_sarsa[0]['num_steps']

def get_mc_agent_info(eps, step_size):
    return  {
            'initializer_params': {'init_method': 'zero'},
            'action_selection_params': {
                'method': 'epsilon_greedy',
                'kwargs': {'eps': eps}
            },
            'step_size': step_size}

onpolicy_mc_agent = nStepSarsaAgent(num_states, num_actions, get_mc_agent_info(0.1, 0.3))
rl_task_onpolicy_mc = RLTask(onpolicy_mc_agent, env, MAX_EPISODE_STEPS)
metrics_onpolicy_mc = run_experiment(rl_task_onpolicy_mc, metric_objs, metric_objs_test, random_seed=50)

def get_nstep_agent_info(eps, step_size, num_steps):
    return  {
            'initializer_params': {'init_method': 'zero'},
            'action_selection_params': {
                'method': 'epsilon_greedy',
                'kwargs': {'eps': eps}
            },
            'step_size': step_size,
            'num_steps': num_steps
            }

metrics_list_nstep_sarsa = []
for n in (2,4):
    nstep_sarsa_agent = nStepSarsaAgent(num_states, num_actions, get_nstep_agent_info(0.1, 0.3, n))
    rl_task_nstep_sarsa = RLTask(nstep_sarsa_agent, env, MAX_EPISODE_STEPS)
    metrics_list_nstep_sarsa.append((n, run_experiment(rl_task_nstep_sarsa, metric_objs, metric_objs_test, random_seed=50)))

def metric_after_x_episodes(metric_name, episode_num, metric_td0, metric_mc, metric_list_nstep, test=False):
    ans = []
    x = []
    print(len(metric_td0[test*1][metric_name]))
    ans.append(metric_td0[test*1][metric_name][episode_num])
    x.append('1')
    for metric in metric_list_nstep:
        ans.append(metric[1][test*1][metric_name][episode_num])
        x.append(str(metric[0]))
    ans.append(metric_mc[test*1][metric_name][episode_num])
    x.append('MC')
    return x, ans

x, y = metric_after_x_episodes('return', 1, metrics_sarsa, metrics_onpolicy_mc, metrics_list_nstep_sarsa)
x, y = metric_after_x_episodes('return', -1, metrics_sarsa, metrics_onpolicy_mc, metrics_list_nstep_sarsa)

rl_task_nstep_sarsa.test_episode([metric_objs_test[1]])['return'][0]


state, _ = env.reset(seed=20)
action = sarsa_agent.select_action(state, greedy=True)
for i in range(MAX_EPISODE_STEPS):
  state, reward, is_terminal, is_truncated, _ = env.step(action)
  action = sarsa_agent.select_action(state, greedy=True)
  if is_terminal or is_truncated:
      break
 
qlearning_agent = nStepQLearningAgent(num_states, num_actions, get_td0_agent_info(0.1, 0.3))
rl_task_qlearning = RLTask(qlearning_agent, env, MAX_EPISODE_STEPS)
metrics_qlearning = run_experiment(rl_task_qlearning, metric_objs, metric_objs_test, random_seed=50)

offpolicy_mc_agent = nStepQLearningAgent(num_states, num_actions, get_mc_agent_info(0.1, 0.3))
rl_task_offpolicy_mc = RLTask(offpolicy_mc_agent, env, MAX_EPISODE_STEPS)
metrics_offpolicy_mc = run_experiment(rl_task_offpolicy_mc, metric_objs, metric_objs_test, random_seed=50)

qsigma1_agent = nStepQLearningAgent(num_states, num_actions, get_nstep_agent_info(0.1, 0.3, 2))
rl_task_qsigma1 = RLTask(qsigma1_agent, env, MAX_EPISODE_STEPS)
metrics = run_experiment(rl_task_qsigma1, metric_objs, metric_objs_test, random_seed=50)
rl_task_qsigma1.test_episode([metric_objs_test[1]])['return'][0]

def get_qsigma0_agent_info(eps, step_size, num_steps):
    return  {
            'initializer_params': {'init_method': 'zero'},
            'action_selection_params': {
                'method': 'epsilon_greedy',
                'kwargs': {'eps': eps}
            },
            'step_size': step_size,
            'num_steps': num_steps,
            'sigma_func': lambda i: 0
            }

qsigma0_agent = nStepQLearningAgent(num_states, num_actions, get_qsigma0_agent_info(0.1, 0.3, 2))
rl_task_qsigma0 = RLTask(qsigma0_agent, env, MAX_EPISODE_STEPS)
metrics = run_experiment(rl_task_qsigma0, metric_objs, metric_objs_test)
rl_task_qsigma0.test_episode([metric_objs_test[1]])['return'][0]