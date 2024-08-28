import numpy as np
from .nstep_td import nStepSarsaAgent, nStepQLearningAgent
from .monte_carlo import MonteCarloAgent
from .td_0 import SarsaAgent, QLearningAgent

def env_dynamics(state, action):

    next_state = state + 2*action - 1

    reward = -1
    done = False
    if next_state in (0,4):
        reward = 0
        done = True

    return next_state, reward, done

def test_monte_carlo_equivalent():
    
    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    }
    agent = nStepSarsaAgent(num_states=5, num_actions=2, agent_info=agent_info)
    agent.reset()

    s = 2
    a = agent.start(s)
    done = False
    t = 0
    while not done:
        assert len(agent.trajectory) == t
        s, r, done = env_dynamics(s, a)
        a = agent.step(s, r)
        t += 1
    assert len(agent.trajectory) == t

    mc_agent = MonteCarloAgent(num_states=5, num_actions=2, agent_info=agent_info)
    agent.reset()
    mc_agent.reset()
    agent.start(2)
    agent.last_action = 1
    agent.trajectory.extend([(-1, 3, 0), (3, 1, 0), (-4, 0, 1), (2, 4, 0)])
    mc_agent.trajectory = [(None, 2, 1), (-1, 3, 0), (3, 1, 0), (-4, 0, 1), (2, 4, 0)]
    agent.on_episode_end()
    mc_agent._learn_from_trajectory()
    print(agent.action_values)
    print(mc_agent.action_values)
    assert((agent.action_values == mc_agent.action_values).all())

def test_sarsa_equivalent():

    agent_info = {
    'initializer_params': {'init_method': 'constant', 'constant': 1},
    'action_selection_params': {'method': 'greedy', 'kwargs': {'tie_break': 'first'}},
    'step_size': 0.2,
    'discount': 0.9,
    'num_steps': 1
    }
    agent = SarsaAgent(5, 3, agent_info)
    nstep_agent = nStepSarsaAgent(5, 3, agent_info)
    agent.reset()
    nstep_agent.reset()
    a1 = agent.start(2)
    a2 = nstep_agent.start(2)
    assert a1==a2
    for _ in range(20):
        r = np.random.random()*20 - 10
        s = np.random.randint(0,5)
        a1 = agent.step(s,r)
        a2 = nstep_agent.step(s,r)
        assert a1 == a2
    assert (agent.action_values == nstep_agent.action_values).all()

def test_qlearning_equivalent():
    
    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'greedy', 'kwargs': {'tie_break': 'first'}},
    'step_size': 0.2,
    'discount': 0.9,
    'num_steps': 1
    }
    agent = QLearningAgent(5, 3, agent_info)
    nstep_agent = nStepQLearningAgent(5, 3, agent_info)
    agent.reset()
    nstep_agent.reset()
    a1 = agent.start(2)
    a2 = nstep_agent.start(2)
    assert a1==a2
    for _ in range(20):
        r = np.random.random()*20 - 10
        s = np.random.randint(0,5)
        a1 = agent.step(s,r)
        nstep_agent.step(s,r)
        r,s,_ = nstep_agent.trajectory[-1]
        nstep_agent.trajectory[-1] = (r,s,a1)
    assert (agent.action_values == nstep_agent.action_values).all()

def test_nstep_sarsa():

    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'greedy', 'kwargs': {'tie_break': 'first'}},
    'step_size': 0.2,
    'discount': 0.9,
    'num_steps': 5
    }

    trajectory = [(-1, 3), (5, 2), (3, 4), (-2, 0), (-5, 2), (10, 1), (1, 0), (-1, 1), (-3, 4)]

    agent = nStepSarsaAgent(5, 3, agent_info)
    agent.reset()
    s = 2
    a = agent.start(s)
    assert a==0
    actions = []
    for i,(r,s) in enumerate(trajectory):
        actions.append(a)
        a = agent.step(s,r)
        assert a==0
        if i < 4:
            assert a == 0
            assert len(agent.trajectory)==i+1
        else:
            assert len(agent.trajectory)==5

    assert np.isclose(agent.action_values[3,0], 1.7992, 10e-8)
    assert np.isclose(agent.action_values[2,0], 1.20986, 10e-8)
    assert agent.action_values[4,0] == 0.33458
    assert round(agent.action_values[0,0], 8) == 0.46205323

    agent.on_episode_end()

    assert round(agent.action_values[2,0], 8) == 2.59239159
    assert round(agent.action_values[0,0], 8) == -0.31615546
    assert round(agent.action_values[1,0], 8) == -0.84903872
    assert agent.action_values[4,0] == 0.33458

def test_nstep_qlearning():

    agent_info = {
    'initializer_params': {'init_method': 'zero'},
    'action_selection_params': {'method': 'epsilon_greedy', 'kwargs': {'eps': 0.3}},
    'step_size': 0.2,
    'discount': 0.9,
    'num_steps': 3
    }

    trajectory = [(-1, 3, 0), (5, 2, 0), (1, 0, 1), (-1, 1, 0), (-3, 4, 0)]

    agent = nStepQLearningAgent(5, 2, agent_info)
    agent.reset()
    s = 2
    a = agent.start(s)
    targets = []
    for i,(r,s,a) in enumerate(trajectory):
        agent.trajectory.append((r,s,a))
        if i >=2:
            if i > 2:
                agent.trajectory.popleft()
            target_q = agent._learn_from_trajectory(update_all=False)
            targets.append(target_q)

    assert round(targets[0], 8) == 5.41522491
    assert round(targets[1], 8) == 6.05882353
    assert round(targets[2], 8) == 1

    agent_info['sigma_func'] = lambda i: 0
    agent = nStepQLearningAgent(5, 2, agent_info)
    agent.reset()
    s = 2
    a = agent.start(s)
    targets = []
    for i,(r,s,a) in enumerate(trajectory):
        agent.trajectory.append((r,s,a))
        if i >=2:
            if i > 2:
                agent.trajectory.popleft()
            target_q = agent._learn_from_trajectory(update_all=False)
            targets.append(target_q)

    assert round(targets[0], 8) == 4.31
    assert round(targets[1], 8) == 5.9
    assert round(targets[2], 8) == 1