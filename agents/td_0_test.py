from .td_0 import *

def test_sarsa():
    agent_info = {
    'initializer_params': {'init_method': 'constant', 'constant': 1},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'step_size': 0.2,
    'discount': 0.9
    }
    agent = SarsaAgent(5, 3, agent_info)
    s0 = 1
    agent.reset()
    a0 = agent.start(s0)
    s1 = 0
    r1 = -1
    a1 = agent.step(s1, r1)
    assert agent.action_values[s0,a0] == 0.78
    s2 = 2
    r2 = -2
    a2 = agent.step(s2, r2)
    assert agent.action_values[s1,a1] == 0.58
    
def test_q_learning():
    q_values = np.array([
        [1, 2, 3],
        [5, 4, 3],
        [1, 10, 8],
        [0, -10, -2],
        [1, 1, 1]])
    agent_info = {
    'initializer_params': {'init_method': 'custom', 'values': q_values},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'step_size': 0.2,
    'discount': 0.9
    }
    agent = QLearningAgent(5, 3, agent_info)
    s0 = 1
    a0 = 1
    agent.reset()
    agent.start(s0)
    agent.last_action = a0
    s1 = 0
    r1 = -1
    agent.step(s1, r1)
    assert agent.action_values[s0,a0] == 3.54
    a1 = 2
    agent.last_action = a1
    s2 = 2
    r2 = -2
    agent.step(s2, r2)
    assert agent.action_values[s1,a1] == 3.8
    a2 = 0
    agent.last_action = a2
    s3 = 0
    r3 = 3
    agent.step(s3, r3)
    assert agent.action_values[s2,a2] == 2.084

def test_expected_sarsa():
    q_values = np.array([
        [1, 2, 3],
        [5, 4, 3],
        [1, 10, 8],
        [0, -10, -2],
        [1, 1, 1]])
    agent_info = {
    'initializer_params': {'init_method': 'custom', 'values': q_values},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'step_size': 0.2,
    'discount': 0.9
    }
    agent = ExpectedSarsaAgent(5, 3, agent_info)
    s0 = 1
    a0 = 1
    agent.reset()
    agent.start(s0)
    agent.last_action = a0
    s1 = 0
    r1 = -1
    agent.step(s1, r1)
    assert np.isclose(agent.action_values[s0,a0], 3.486)
    a1 = 2
    agent.last_action = a1
    s2 = 2
    r2 = -2
    agent.step(s2, r2)
    assert agent.action_values[s1,a1] == 3.602
    a2 = 0
    agent.last_action = a2
    s3 = 0
    r3 = 3
    agent.step(s3, r3)
    assert agent.action_values[s2,a2] == 1.972688

def test_double_q_learning():
    q_values_1 = np.array([
        [1, 2, 3],
        [5, 4, 3],
        [1, 10, 8],
        [0, -10, -2],
        [1, 1, 1]], dtype=float)
    q_values_2 = np.array([
        [3, 2, 1],
        [3, 4, 5],
        [8, 10, 1],
        [-2, -10, 0],
        [1, 1, 1]], dtype=float)
    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'step_size': 0.2,
    'discount': 0.9
    }
    agent = DoubleQLearningAgent(5, 3, agent_info)
    s0 = 1
    a0 = 1
    agent.reset()
    agent.start(s0)
    agent.last_action = a0
    agent.action_values_1 = q_values_1
    agent.action_values_2 = q_values_2
    s1 = 0
    r1 = -1
    agent.step(s1, r1)
    assert agent.action_values[s0,a0] == 3.59
    a1 = 2
    agent.last_action = a1
    s2 = 2
    r2 = -2
    agent.step(s2, r2)
    assert agent.action_values[s1,a1] in (2.4, 2.6)
    a2 = 0
    agent.last_action = a2
    s3 = 0
    r3 = 3
    agent.step(s3, r3)
    assert agent.action_values[s2,a2] in (4.79, 4.898, 4.09)