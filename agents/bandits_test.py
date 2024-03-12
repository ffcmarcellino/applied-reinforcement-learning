import numpy as np
from .bandits import BaseAgent, KBandits, GradientBandits

def test_select_action():

    agent_info = {
        'initializer_params': {'init_method': 'custom', 'values': np.array([[-2, 6, 11, 2, 0]])},
        'step_size': '1/n'
    }

    # test greedy action selection
    agent_info['action_selection_params'] = {'method': 'greedy'}
    agent = BaseAgent(1, 5, agent_info)
    agent.reset()
    agent.start(0)

    action_count = np.zeros(5)
    for i in range(10):
        action = agent.select_action(0)
        action_count[action] += 1
    assert action_count[2] == 10

    # test epsilon-greedy action selection
    agent_info['action_selection_params'] = {'method': 'epsilon_greedy'}
    agent_info['action_selection_params']['kwargs'] = {'eps': 0.2}
    agent = BaseAgent(1, 5, agent_info)
    agent.reset()
    agent.start(0)
    action_count = np.zeros(5)
    for i in range(200):
        action = agent.select_action(0)
        action_count[action] += 1
    assert all([action_count[2] > action_count[a] for a in range(5) if a!= 2])

    # test UCB action selection
    agent_info['initializer_params'] = {'init_method': 'zero'}
    agent_info['action_selection_params'] = {'method': 'ucb'}
    agent_info['action_selection_params']['kwargs'] = {'c': 10}
    agent = BaseAgent(1, 5, agent_info)
    agent.reset()
    agent.start(0)
    agent.num_visits = np.array([[1, 10, 0, 3, 9]])
    action_count = np.zeros(5)
    for i in range(10):
        action = agent.select_action(0)
        action_count[action] += 1
    assert action_count[2] == 10

    # test softmax action selection
    agent_info['initializer_params'] = {'init_method': 'custom', 'values': np.array([[1, 3, 2]])}
    agent_info['action_selection_params'] = {'method': 'softmax'}
    agent = BaseAgent(1, 3, agent_info)
    agent.reset()
    agent.start(0)
    action_count = np.zeros(3)
    for i in range(200):
        action = agent.select_action(0)
        action_count[action] += 1
    assert action_count[1] > action_count[2]
    assert action_count[2] > action_count[0]

def test_k_bandits():

    agent_info = {
        'initializer_params': {'init_method': 'zero'},
        'action_selection_params': {'method': 'greedy'},
        'step_size': '1/n'
    }

    agent = KBandits(3, 5, agent_info)

    agent.reset()
    agent.start(0)

    agent.step(1, -10)
    assert agent.num_visits[0, 0] == 1
    assert agent.action_values[0, 0] == -10
    assert agent.last_state == 1
    assert agent.last_action == 0

    agent.step(0, -3)
    assert agent.num_visits[1, 0] == 1
    assert agent.action_values[1, 0] == -3
    assert agent.last_state == 0
    assert agent.last_action == 1

    agent.step(0, 3)
    assert agent.num_visits[0, 1] == 1
    assert agent.action_values[0, 1] == 3
    assert agent.last_state == 0
    assert agent.last_action == 1

    agent.step(1, -4)
    assert agent.num_visits[0, 1] == 2
    assert agent.action_values[0, 1] == -1/2
    assert agent.last_state == 1
    assert agent.last_action == 1

def test_gradient_bandits():

    agent_info = {
        'initializer_params': {'init_method': 'zero'},
        'step_size': 0.1
    }

    agent = GradientBandits(3, 5, agent_info)

    agent.reset()
    agent.start(0)

    action = agent.last_action
    agent.step(1, -10)
    assert agent.num_visits[0, action] == 1
    assert agent.action_values[0, action] == 1.0/5 - 1
    assert all([agent.action_values[0, a] == 1.0/5 for a in range(5) if a != action])
    assert agent.avg_reward[0] == -10
    assert agent.last_state == 1
