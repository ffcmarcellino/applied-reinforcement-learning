import numpy as np
from envs.tic_tac_toe_env.tic_tac_toe import TicTacToe
from agents.monte_carlo import MonteCarloAgent
from rl_task import RLTask
from visualization import *

num_actions = 9
metrics_info = {'perc_pair_visits', 'cum_reward'}
NUM_EPISODES_TRAIN = 10
NUM_EPISODES_TEST = 10
step_size = '1/n'
plot_precision = 2

def update_result_cnt(reward, trunc_cnt, draw_cnt, win_cnt, loss_cnt):

    if reward == TicTacToe.REWARD_INVALID:
        trunc_cnt += 1
    elif reward == TicTacToe.REWARD_DRAW:
        draw_cnt += 1
    elif reward == TicTacToe.REWARD_WIN:
        win_cnt += 1
    elif reward == TicTacToe.REWARD_LOSS:
        loss_cnt += 1
    return trunc_cnt, draw_cnt, win_cnt, loss_cnt

def episode_test(env, agent, return_seq = False, render=False, save=False):
    states, boards, actions = [], [], []
    state, _ = env.reset()
    if render: env.render(save=save)
    action = agent.select_action(state)
    if return_seq:
        states.append(state)
        boards.append(env.board.copy())
        actions.append(action)
    done = False
    while not done:
        state, reward, is_terminal, is_truncated, _ = env.step(action)
        if render: env.render(save=save)
        action = agent.select_action(state)
        done = is_terminal or is_truncated
        if return_seq:
            states.append(state)
            boards.append(env.board.copy())
            actions.append(action)
    if return_seq:
        return reward, states, boards, actions
    return reward

def episode_test_optimal(env):

    env.reset()
    done = False
    while not done:
        x,y = env.optimal_policy(env.board, player=1)
        action = env.pos_to_action(x,y)
        _, reward, is_terminal, is_truncated, _ = env.step(action)
        done = is_terminal or is_truncated

    return reward

ttt_random_es = TicTacToe(exploring_starts=True)
ttt_random = TicTacToe()
num_states = max(ttt_random_es.obs_to_state_map.values()) + 1
allowed_actions_mask = ttt_random.get_allowed_actions_mask()

agent_info_es = {
    'initializer_params': {'init_method': 'constant', 'constant': 1},
    'action_selection_params': {
        'method': 'greedy',
        'allowed_actions_mask': allowed_actions_mask,
        'kwargs': {'tie_break': 'random'}
    },
    'step_size': step_size,
    'exploring_starts': True
}

mc_agent_es = MonteCarloAgent(num_states, num_actions, agent_info_es)

rl_task_random_es = RLTask(mc_agent_es, ttt_random_es)

rl_task_random_es.agent.reset(0, hard_reset=True)

trunc_cnt_test, draw_cnt_test, win_cnt_test, loss_cnt_test = 0, 0, 0, 0
for i in range(NUM_EPISODES_TEST):
    final_reward = episode_test(ttt_random, rl_task_random_es.agent)
    trunc_cnt_test, draw_cnt_test, win_cnt_test, loss_cnt_test = update_result_cnt(final_reward, trunc_cnt_test, draw_cnt_test, win_cnt_test, loss_cnt_test)

for i in range(plot_precision):

    action_values_prev = rl_task_random_es.agent.action_values.copy()

    for j in range(NUM_EPISODES_TRAIN // plot_precision):
        metrics = rl_task_random_es.run_episode(metrics_info)

    np.abs(action_values_prev-rl_task_random_es.agent.action_values).max()
    metrics['perc_pair_visits']

for k in range(NUM_EPISODES_TEST):
    final_reward = episode_test_optimal(ttt_random)

num_after_states = max(ttt_random.obs_to_after_state_map.values()) + 1
state_action_to_after_state_map = ttt_random.get_state_action_to_after_state_map(rotation_invariant=True)

agent_info_es_as = {
    'initializer_params': {'init_method': 'constant', 'constant': 1},
    'action_selection_params': {
        'method': 'greedy',
        'allowed_actions_mask': allowed_actions_mask,
        'kwargs': {'tie_break': 'random'}
    },
    'step_size': step_size,
    'exploring_starts': True,
    'state_action_to_after_state_map': state_action_to_after_state_map
}

mc_agent_es_as = MonteCarloAgent(num_states, num_actions, agent_info_es_as)

rl_task_random_es_as = RLTask(mc_agent_es_as, ttt_random_es)

rl_task_random_es_as.agent.reset(0, hard_reset=True)

for i in range(NUM_EPISODES_TEST):
    final_reward = episode_test(ttt_random, rl_task_random_es_as.agent)

for i in range(plot_precision):

    action_values_prev = rl_task_random_es_as.agent.action_values.copy()

    for j in range(NUM_EPISODES_TRAIN // plot_precision):
        metrics = rl_task_random_es_as.run_episode(metrics_info)

    np.abs(action_values_prev-rl_task_random_es_as.agent.action_values).max()
    metrics['perc_pair_visits']

agent_info_sp_as = {
    'initializer_params': {'init_method': 'constant', 'constant': 1},
    'action_selection_params': {
        'method': 'epsilon_greedy',
        'allowed_actions_mask': allowed_actions_mask,
        'kwargs': {'eps': 0.3}
    },
    'step_size': step_size,
    'state_action_to_after_state_map': state_action_to_after_state_map
}

mc_agent_sp_as = MonteCarloAgent(num_states, num_actions, agent_info_sp_as)

rl_task_random_sp_as = RLTask(mc_agent_sp_as, ttt_random)

rl_task_random_sp_as.agent.reset(0, hard_reset=True)
get_soft_policy = rl_task_random_sp_as.agent.get_policy

rl_task_random_sp_as.agent.get_policy = rl_task_random_sp_as.agent._get_policy_method('greedy', tie_break='random')
for i in range(NUM_EPISODES_TEST):
    final_reward = episode_test(ttt_random, rl_task_random_sp_as.agent)
rl_task_random_sp_as.agent.get_policy = get_soft_policy

for i in range(plot_precision):

    action_values_prev = rl_task_random_sp_as.agent.action_values.copy()

    for j in range(NUM_EPISODES_TRAIN // plot_precision):
        metrics = rl_task_random_sp_as.run_episode(metrics_info)

    np.abs(action_values_prev-rl_task_random_sp_as.agent.action_values).max()
    metrics['perc_pair_visits']

ttt_random.obs_to_state_map[ttt_random._encode_board(np.array([[0,0,-1],[0,0,0],[0,0,0]]))]

ttt_optimal = TicTacToe(adversary_policy='optimal')

rl_task_random_sp_as.agent.get_policy = rl_task_random_sp_as.agent._get_policy_method('greedy', tie_break='random')
for k in range(NUM_EPISODES_TEST):
    final_reward = episode_test(ttt_optimal, rl_task_random_sp_as.agent)
rl_task_random_sp_as.agent.get_policy = get_soft_policy

agent_info_sp_as_opt = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {
        'method': 'epsilon_greedy',
        'allowed_actions_mask': allowed_actions_mask,
        'kwargs': {'eps': 0.3}
    },
    'step_size': step_size,
    'state_action_to_after_state_map': state_action_to_after_state_map
}

mc_agent_sp_as_opt = MonteCarloAgent(num_states, num_actions, agent_info_sp_as_opt)

rl_task_optimal_sp_as = RLTask(mc_agent_sp_as_opt, ttt_optimal)

rl_task_optimal_sp_as.agent.reset(0, hard_reset=True)
get_soft_policy = rl_task_optimal_sp_as.agent.get_policy

rl_task_optimal_sp_as.agent.get_policy = rl_task_optimal_sp_as.agent._get_policy_method('greedy', tie_break='random')
for i in range(NUM_EPISODES_TEST):
    final_reward = episode_test(ttt_optimal, rl_task_optimal_sp_as.agent)
rl_task_optimal_sp_as.agent.get_policy = get_soft_policy

for i in range(plot_precision):

    action_values_prev = rl_task_optimal_sp_as.agent.action_values.copy()

    for j in range(NUM_EPISODES_TRAIN // plot_precision):
        metrics = rl_task_optimal_sp_as.run_episode(metrics_info)

    np.abs(action_values_prev-rl_task_optimal_sp_as.agent.action_values).max()
    metrics['perc_pair_visits']
