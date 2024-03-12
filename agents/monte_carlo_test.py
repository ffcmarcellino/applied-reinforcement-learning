import numpy as np
from .monte_carlo import MonteCarloAgent

def env_dynamics(state, action):

    next_state = state + 2*action - 1

    reward = -1
    done = False
    if next_state in (0,4):
        reward = 0
        done = True

    return next_state, reward, done

def test_on_policy_learning():
    
    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'greedy'}
    }
    agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)

    trajectory = [(None, 2, 1), (-1, 3, 1), (-1, 4, 1)]
    agent.reset()
    agent.start(0)
    agent.trajectory = trajectory
    agent._learn_from_trajectory()
    assert (agent.action_values == np.array([[0, 0], [0, 0], [0, -2], [0, -1], [0, 0]])).all()

def test_on_policy_learning_after_state():
    
    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'greedy'},
    'state_action_to_after_state_map': np.array([[0, 0], [1, 2], [3, 3], [2, 1], [0, 0]])
    }
    agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)
    agent.reset()
    agent.start(0)
    agent.trajectory = [(None, 2, 1), (-1, 3, 0), (-1, 2, 0), (-1, 1, 1), (-1, 2, 1), (-1, 3, 1), (-1, 4, 1)]
    agent._learn_from_trajectory()
    assert (agent.action_values == np.array([[0, 0], [-1, -5], [-6, -6], [-5, -1], [0, 0]])).all()
    agent.trajectory = [(None, 2, 0), (-1, 1, 1), (-1, 2, 1), (-1, 3, 0), (-1, 2, 0), (-1, 1, 0), (-2, 0, 1)]
    agent._learn_from_trajectory()
    assert (agent.action_values == np.array([[0, 0], [-1.5, -5.5], [-6.5, -6.5], [-5.5, -1.5], [0, 0]])).all()

def test_ordinary_importance_sampling():

    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'learning_params': {'method': 'off_policy', 'type': 'importance_sampling', 'average_type': 'ordinary'}
    }
    agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)
    agent.reset()
    agent.start(0)
    agent.trajectory = [(None, 2, 1), (1, 3, 0), (1, 2, 0), (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),  (1, 0, 0),  (1, 1, 0)]
    agent._learn_from_trajectory()
    assert (agent.action_values == np.array([[1, 0], [0, 5/(0.85**4)], [6/(0.85**5), 2/(0.85**3) + 4/(0.85**7)], [7/(0.85**6), 3/(0.85**2)], [0, 2/0.85]])).all()
    agent.trajectory = [(None, 2, 0), (-1, 1, 1), (-1, 2, 1), (-1, 3, 0), (-1, 2, 0), (-1, 1, 0), (-2, 0, 1)]
    agent._learn_from_trajectory()
    assert (agent.action_values == np.array([[1, 0], [-2, 5/(0.85**4)], [6/(0.85**5), 2/(0.85**3) + 4/(0.85**7)], [7/(0.85**6), 3/(0.85**2)], [0, 2/0.85]])).all()

def test_weighted_importance_sampling():

    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'learning_params': {'method': 'off_policy', 'type': 'importance_sampling', 'average_type': 'weighted'}
    }
    agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)
    agent.reset()
    agent.start(0)
    agent.trajectory = [(None, 2, 1), (1, 3, 0), (1, 2, 0), (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),  (1, 0, 0),  (1, 1, 0)]
    agent._learn_from_trajectory()
    assert np.isclose(agent.action_values, np.array([[1, 0], [0, 5], [6, (4/(0.85**3) + 8/(0.85**7))/(1/(0.85**3) + 1/(0.85**7))], [7, 3], [0, 2]]), atol=1e-8).all()
    agent.trajectory = [(None, 2, 0), (-1, 1, 1), (-1, 2, 1), (-1, 3, 0), (-1, 2, 0), (-1, 1, 0), (-2, 0, 1)]
    agent._learn_from_trajectory()
    assert np.isclose(agent.action_values, np.array([[1, 0], [-2, 5], [6, (4/(0.85**3) + 8/(0.85**7))/(1/(0.85**3) + 1/(0.85**7))], [7, 3], [0, 2]]), atol=1e-8).all()

def test_discounting_aware_ordinary_importance_sampling():
    
    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'learning_params': {'method': 'off_policy', 'type': 'discounting_aware_importance_sampling', 'average_type': 'ordinary'},
    'discount': 0.9
    }
    agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)
    agent.reset()
    agent.start(0)
    agent.trajectory = [(None, 2, 1), (1, 3, 0), (1, 2, 0), (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),  (1, 0, 0),  (1, 1, 0)]
    agent._learn_from_trajectory()
    assert np.isclose(agent.action_values, np.array([[1, 0], [0, 0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 5*(0.9/0.85)**4], [0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 6*(0.9/0.85)**5, 0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*2.2/(0.85**3) + 0.25*(0.9/0.85)**4 + 0.3*(0.9/0.85)**5 + 0.35*(0.9/0.85)**6 + 4*(0.9/0.85)**7], [0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 0.6*(0.9/0.85)**5 + 7*(0.9/0.85)**6, 0.1 + 0.18/0.85 + 2.43/(0.85**2)], [0, 0.1 + 1.8/0.85]]), atol=1e-8).all()
    agent.trajectory = [(None, 3, 0), (-1, 2, 0), (-1, 1, 0), (-2, 0, 1)]
    agent._learn_from_trajectory()
    assert np.isclose(agent.action_values, np.array([[1, 0], [-2, 0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 5*(0.9/0.85)**4], [0.09/0.85 + 0.1215/(0.85**2) + 0.729*0.2/(0.85**3) + 0.25*(0.9/0.85)**4 + 3*(0.9/0.85)**5, 0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*2.2/(0.85**3) + 0.25*(0.9/0.85)**4 + 0.3*(0.9/0.85)**5 + 0.35*(0.9/0.85)**6 + 4*(0.9/0.85)**7], [0.09/0.85 + 0.1215/(0.85**2) + 0.729*0.2/(0.85**3) + 0.25*(0.9/0.85)**4 + 0.3*(0.9/0.85)**5 + 3.5 *(0.9/0.85)**6, 0.1 + 0.18/0.85 + 2.43/(0.85**2)], [0, 0.1 + 1.8/0.85]]), atol=1e-8).all()

def test_discounting_aware_weighted_importance_sampling():
    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'learning_params': {'method': 'off_policy', 'type': 'discounting_aware_importance_sampling', 'average_type': 'weighted'},
    'discount': 0.9
    }
    agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)
    agent.reset()
    agent.start(0)
    agent.trajectory = [(None, 2, 1), (1, 3, 0), (1, 2, 0), (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),  (1, 0, 0),  (1, 1, 0)]
    agent._learn_from_trajectory()
    print(agent.action_values)
    print(np.array([[1, 0], [0, (0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 5*(0.9/0.85)**4) / (0.1 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + (0.9/0.85)**4)], [(0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 6*(0.9/0.85)**5) / (0.1 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + 0.1*(0.9/0.85)**4 + (0.9/0.85)**5), (0.2 + 0.36/0.85 + 0.486/(0.85**2) + 0.729*4.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 0.6*(0.9/0.85)**5 + 0.7*(0.9/0.85)**6 + 8*(0.9/0.85)**7) / (0.2 + 0.18/0.85 + 0.162/(0.85**2) + 0.729*1.1/(0.85**3) + 0.1*(0.9/0.85)**4 + 0.1*(0.9/0.85)**5 + 0.1*(0.9/0.85)**6 + (0.9/0.85)**7)], [(0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 0.6*(0.9/0.85)**5 + 7*(0.9/0.85)**6) / (0.1 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + 0.1*(0.9/0.85)**4 + 0.1*(0.9/0.85)**5 + (0.9/0.85)**6), (0.1 + 0.18/0.85 + 2.43/(0.85**2)) / (0.1 + 0.09/0.85 + 0.81/(0.85**2))], [0, (0.1 + 1.8/0.85) / (0.1 + 0.9/0.85)]]))
    assert np.isclose(agent.action_values, np.array([[1, 0], [0, (0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 5*(0.9/0.85)**4) / (0.1 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + (0.9/0.85)**4)], [(0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 6*(0.9/0.85)**5) / (0.1 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + 0.1*(0.9/0.85)**4 + (0.9/0.85)**5), (0.2 + 0.36/0.85 + 0.486/(0.85**2) + 0.729*4.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 0.6*(0.9/0.85)**5 + 0.7*(0.9/0.85)**6 + 8*(0.9/0.85)**7) / (0.2 + 0.18/0.85 + 0.162/(0.85**2) + 0.729*1.1/(0.85**3) + 0.1*(0.9/0.85)**4 + 0.1*(0.9/0.85)**5 + 0.1*(0.9/0.85)**6 + (0.9/0.85)**7)], [(0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 0.6*(0.9/0.85)**5 + 7*(0.9/0.85)**6) / (0.1 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + 0.1*(0.9/0.85)**4 + 0.1*(0.9/0.85)**5 + (0.9/0.85)**6), (0.1 + 0.18/0.85 + 2.43/(0.85**2)) / (0.1 + 0.09/0.85 + 0.81/(0.85**2))], [0, (0.1 + 1.8/0.85) / (0.1 + 0.9/0.85)]]), atol=1e-8).all()
    agent.trajectory = [(None, 3, 0), (-1, 2, 0), (-1, 1, 0), (-2, 0, 1)]
    agent._learn_from_trajectory()
    assert np.isclose(agent.action_values, np.array([[1, 0], [-2, (0.1 + 0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 5*(0.9/0.85)**4) / (0.1 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + (0.9/0.85)**4)], [(0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 6*(0.9/0.85)**5) / (0.2 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + 0.1*(0.9/0.85)**4 + (0.9/0.85)**5), (0.2 + 0.36/0.85 + 0.486/(0.85**2) + 0.729*4.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 0.6*(0.9/0.85)**5 + 0.7*(0.9/0.85)**6 + 8*(0.9/0.85)**7) / (0.2 + 0.18/0.85 + 0.162/(0.85**2) + 0.729*1.1/(0.85**3) + 0.1*(0.9/0.85)**4 + 0.1*(0.9/0.85)**5 + 0.1*(0.9/0.85)**6 + (0.9/0.85)**7)], [(0.18/0.85 + 0.243/(0.85**2) + 0.729*0.4/(0.85**3) + 0.5*(0.9/0.85)**4 + 0.6*(0.9/0.85)**5 + 7*(0.9/0.85)**6) / (0.2 + 0.09/0.85 + 0.081/(0.85**2) + 0.729*0.1/(0.85**3) + 0.1*(0.9/0.85)**4 + 0.1*(0.9/0.85)**5 + (0.9/0.85)**6), (0.1 + 0.18/0.85 + 2.43/(0.85**2)) / (0.1 + 0.09/0.85 + 0.81/(0.85**2))], [0, (0.1 + 1.8/0.85) / (0.1 + 0.9/0.85)]]), atol=1e-8).all()

def test_per_decision_importance_sampling():

    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'learning_params': {'method': 'off_policy', 'type': 'per_decision_importance_sampling'}
    }
    agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)
    agent.reset()
    agent.start(0)
    agent.trajectory = [(None, 2, 1), (1, 3, 0), (1, 2, 0), (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),  (1, 0, 0),  (1, 1, 0)]
    agent._learn_from_trajectory()
    assert np.isclose(agent.action_values, np.array([[1, 0], [0, 1 + 1/0.85 + 1/(0.85**2) + 1/(0.85**3) + 1/(0.85**4)], [1 + 1/0.85 + 1/(0.85**2) + 1/(0.85**3) + 1/(0.85**4) + 1/(0.85**5), 1 + 1/0.85 + 1/(0.85**2) + 1/(0.85**3) + 0.5/(0.85**4) + 0.5/(0.85**5) + 0.5/(0.85**6) + 0.5/(0.85**7)], [1 + 1/0.85 + 1/(0.85**2) + 1/(0.85**3) + 1/(0.85**4) + 1/(0.85**5) + 1/(0.85**6), 1 + 1/0.85 + 1/(0.85**2)], [0, 1 + 1/0.85]]), atol=1e-8).all()
    agent.trajectory = [(None, 3, 0), (-1, 2, 0), (-1, 1, 0), (-2, 0, 1)]
    agent._learn_from_trajectory()
    assert np.isclose(agent.action_values, np.array([[1, 0], [-2, 1 + 1/0.85 + 1/(0.85**2) + 1/(0.85**3) + 1/(0.85**4)], [0.5/0.85 + 0.5/(0.85**2) + 0.5/(0.85**3) + 0.5/(0.85**4) + 0.5/(0.85**5), 1 + 1/0.85 + 1/(0.85**2) + 1/(0.85**3) + 0.5/(0.85**4) + 0.5/(0.85**5) + 0.5/(0.85**6) + 0.5/(0.85**7)], [0.5/0.85 + 0.5/(0.85**2) + 0.5/(0.85**3) + 0.5/(0.85**4) + 0.5/(0.85**5) + 0.5/(0.85**6), 1 + 1/0.85 + 1/(0.85**2)], [0, 1 + 1/0.85]]), atol=1e-8).all()

def test_monte_carlo():
    
    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'greedy'}
    }
    agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)
        
    # Fisrt pass

    done = False
    s = 2
    agent.reset()
    a = agent.start(s)
    while not done:
        s, r, done = env_dynamics(s, a)
        agent.step(s, r)

    agent._learn_from_trajectory()

    first_action_values = agent.action_values.copy()
    first_num_visits = agent.num_visits.copy()
    assert agent.trajectory[0] == (None, 2, a)
    assert agent.trajectory[-1][:2] in ((0, 0), (0, 4))
    assert first_action_values[2][a] < 0
    assert first_action_values.min() == first_action_values[2][a]

    # Second pass

    done = False
    s = 2
    a = agent.start(s)
    assert (agent.action_values == first_action_values).all()
    assert (agent.action_values != 0).any()
    while not done:
        s, r, done = env_dynamics(s, a)
        agent.step(s, r)

    agent._learn_from_trajectory()

    assert (first_action_values != agent.action_values).any()
    assert (agent.num_visits >= first_num_visits).all()

    agent.reset()
    agent.start(2)
    assert (agent.action_values == 0).all()
    assert (agent.num_visits == 0).all()

    agent.exploring_starts = True
    action_cnt = np.zeros(agent.num_actions)
    for _ in range(10*agent.num_actions):
        a = agent.start(np.random.randint(0, agent.num_states))
        action_cnt[a] += 1
    assert (action_cnt > 0).all()

    agent2 = MonteCarloAgent(num_states=3, num_actions=5, agent_info=agent_info)
    agent2.exploring_starts = True
    agent2.allowed_actions_mask = np.array([
        [True, True, False, False, True],
        [False, True, True, True, False],
        [True, False, False, True, True]])
    state_action_cnt = np.zeros(agent2.allowed_actions_mask.shape)
    for _ in range(10*agent2.num_actions):
        s = np.random.randint(0, agent2.num_states)
        a = agent2.start(s)
        state_action_cnt[s][a] += 1
    assert (state_action_cnt[agent2.allowed_actions_mask] > 0).all()
    assert (state_action_cnt[~agent2.allowed_actions_mask] == 0).all()