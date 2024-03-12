import numpy as np
from envs.tic_tac_toe_env.tic_tac_toe import TicTacToe
from agents.monte_carlo import MonteCarloAgent
from rl_task import RLTask
from metrics import PercPairVisitsMetric, RewardMetric
from visualization import *

NUM_EPISODES_TRAIN = 10
NUM_EPISODES_TEST = 10
plot_precision = 2
metric_objs = [PercPairVisitsMetric('on_episode_end'), RewardMetric('on_episode_end')]
metric_objs_test = [RewardMetric('on_episode_test_end')]
num_actions = 9
step_size = '1/n'

def update_result_cnt(reward, draw_cnt, win_cnt, loss_cnt):
    
        if reward == TicTacToe.REWARD_DRAW:
            return draw_cnt + 1, win_cnt, loss_cnt
    
        if reward == TicTacToe.REWARD_WIN:
            return draw_cnt, win_cnt + 1, loss_cnt
    
        if reward == TicTacToe.REWARD_LOSS:
            return draw_cnt, win_cnt, loss_cnt + 1
        
def episode_test_optimal(env):

        env.reset()
        done = False
        while not done:
            x,y = env.optimal_policy(env.board, player=1)
            action = env.pos_to_action(x,y)
            _, reward, is_terminal, is_truncated, _ = env.step(action)
            done = is_terminal or is_truncated

        return reward

def run_experiment(rl_task, test_env=None, truncate=False):

    perc_pair_visits = []
    episodes = []  
    draws, wins, losses = [], [], []
    draw_perc, win_perc, loss_perc = [], [], []
    max_deltas = []
    draw_cnt, win_cnt, loss_cnt = 0,0,0

    rl_task.agent.reset()
    rl_task.agent.start(0)

    draw_cnt_test, win_cnt_test, loss_cnt_test = 0, 0, 0
    for i in range(NUM_EPISODES_TEST):
        metrics_test = rl_task.test_episode(metric_objs_test, test_env)
        draw_cnt_test, win_cnt_test, loss_cnt_test = update_result_cnt(metrics_test['reward'][0], draw_cnt_test, win_cnt_test, loss_cnt_test)

    episodes.append(0)
    perc_pair_visits.append(0)

    draw_perc.append(draw_cnt_test*100/NUM_EPISODES_TEST)
    win_perc.append(win_cnt_test*100/NUM_EPISODES_TEST)
    loss_perc.append(loss_cnt_test*100/NUM_EPISODES_TEST)

    draws.append(draw_cnt)
    wins.append(win_cnt)
    losses.append(loss_cnt)
    
    flag = 0
   
    for i in range(plot_precision):

        action_values_prev = rl_task.agent.action_values.copy()
        
        if flag != -1:
            for j in range(NUM_EPISODES_TRAIN // plot_precision):
                metrics = rl_task.run_episode(metric_objs)
                draw_cnt, win_cnt, loss_cnt = update_result_cnt(metrics['reward'][0], draw_cnt, win_cnt, loss_cnt)

        draw_cnt_test, win_cnt_test, loss_cnt_test = 0, 0, 0
        for k in range(NUM_EPISODES_TEST):
            metrics_test = rl_task.test_episode(metric_objs_test, test_env)
            draw_cnt_test, win_cnt_test, loss_cnt_test = update_result_cnt(metrics_test['reward'][0], draw_cnt_test, win_cnt_test, loss_cnt_test)

        max_delta = np.abs(action_values_prev-rl_task.agent.action_values).max()

        max_deltas.append(max_delta)

        episodes.append((i+1)*(NUM_EPISODES_TRAIN // plot_precision))
        perc_pair_visits.append(metrics['perc_pair_visits']*100)

        draw_perc.append(draw_cnt_test*100/NUM_EPISODES_TEST)
        win_perc.append(win_cnt_test*100/NUM_EPISODES_TEST)
        loss_perc.append(loss_cnt_test*100/NUM_EPISODES_TEST)

        draws.append(draw_cnt)
        wins.append(win_cnt)
        losses.append(loss_cnt)
        
        if loss_cnt_test == 0 and flag != -1 and truncate:
            if i == flag + 1:
                flag = -1
            else:
                flag = i
        
    return episodes, perc_pair_visits, max_deltas, draw_perc, win_perc, loss_perc, draws, wins, losses
        
def test_tictactoe_notebook():
    
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

    _,_,_,_,_,_,_,_,_ = run_experiment(rl_task_random_es, ttt_random)

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
    
    _,_,_,_,_,_,_,_,_ = run_experiment(rl_task_random_es_as, ttt_random)

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

    _,_,_,_,_,_,_,_,_ = run_experiment(rl_task_random_sp_as, truncate=True)

    ttt_random.obs_to_state_map[ttt_random._encode_board(np.array([[0,0,-1],[0,0,0],[0,0,0]]))]

    ttt_optimal = TicTacToe(adversary_policy='optimal')
    
    for k in range(NUM_EPISODES_TEST):
        rl_task_random_sp_as.test_episode(metric_objs_test, ttt_optimal)

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

    _,_,_,_,_,_,_,_,_ = run_experiment(rl_task_optimal_sp_as)
