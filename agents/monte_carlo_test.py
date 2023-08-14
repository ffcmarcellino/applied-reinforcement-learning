import numpy as np
from .monte_carlo import MonteCarloAgent

agent_info = {
'initializer_params': {'init_method': 'zero'},
'action_selection_params': {'method': 'greedy'},
'step_size': '1/n'
}
agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)

def env_dynamics(state, action):

    next_state = state + 2*action - 1

    reward = -1
    done = False
    if next_state in (0,4):
        reward = 0
        done = True

    return next_state, reward, done

def test_learn_from_trajectory():

    trajectory = [(None, 2, 1), (-1, 3, 1), (-1, 4, 1)]
    agent.reset(0, hard_reset=True)
    agent.trajectory = trajectory
    agent._learn_from_trajectory()
    assert (agent.action_values == np.array([[0, 0], [0, 0], [0, -2], [0, -1], [0, 0]])).all()

def test_learn_from_trajectory_after_state():

    agent.reset(0, hard_reset=True)
    agent.trajectory = [(None, 2, 1), (-1, 3, 0), (-1, 2, 0), (-1, 1, 1), (-1, 2, 1), (-1, 3, 1), (-1, 4, 1)]
    agent.state_action_to_after_state_map = np.array([[0, 0], [1, 2], [3, 3], [2, 1], [0, 0]])
    agent._learn_from_trajectory_after_state()
    assert (agent.action_values == np.array([[0, 0], [-1, -5], [-6, -6], [-5, -1], [0, 0]])).all()
    agent.trajectory = [(None, 2, 0), (-1, 1, 1), (-1, 2, 1), (-1, 3, 0), (-1, 2, 0), (-1, 1, 0), (-2, 0, 1)]
    agent._learn_from_trajectory_after_state()
    assert (agent.action_values == np.array([[0, 0], [-1.5, -5.5], [-6.5, -6.5], [-5.5, -1.5], [0, 0]])).all()

def test_monte_carlo():

    # Fisrt pass

    done = False
    s = 2
    a = agent.reset(init_state=s, hard_reset=True)
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
    a = agent.reset(init_state=s)
    assert (agent.action_values == first_action_values).all()
    assert (agent.action_values != 0).any()
    while not done:
        s, r, done = env_dynamics(s, a)
        agent.step(s, r)

    agent._learn_from_trajectory()

    assert (first_action_values != agent.action_values).any()
    assert (agent.num_visits >= first_num_visits).all()

    agent.reset(2, hard_reset=True)
    assert (agent.action_values == 0).all()
    assert (agent.num_visits == 0).all()

    agent.exploring_starts = True
    action_cnt = np.zeros(agent.num_actions)
    for _ in range(10*agent.num_actions):
        a = agent.reset(np.random.randint(0, agent.num_states))
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
        a = agent2.reset(s)
        state_action_cnt[s][a] += 1
    assert (state_action_cnt[agent2.allowed_actions_mask] > 0).all()
    assert (state_action_cnt[~agent2.allowed_actions_mask] == 0).all()