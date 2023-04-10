import numpy as np
from .action_selection import *

def test_policy_action_selection():

    # test deterministic policy
    policy = np.array([0, 0, 1, 0, 0])
    action_count = np.zeros(5)
    for i in range(10):
        action = policy_action_selection(policy)
        action_count[action] += 1
    assert action_count[2] == 10

    # test stochastic policy
    policy = np.array([0.2, 0.2, 0, 0.3, 0.3])
    action_count = np.zeros(5)
    for i in range(100):
        action = policy_action_selection(policy)
        action_count[action] += 1
    assert (action_count[2] == 0) and (action_count[action_count != 0] > 0).all()

    policy = np.array([0.9, 0, 0.1])
    action_count = np.zeros(3)
    for i in range(100):
        action = policy_action_selection(policy)
        action_count[action] += 1

    assert (action_count[1] == 0) and (action_count[0] > action_count[2])

def test_greedy_policy():

    # 1D without ties
    action_values = np.array([-10, 3, -2, 11, 0])
    assert greedy_policy(action_values) == 3
    assert greedy_policy(action_values, tie_break='last') == 3
    assert (greedy_policy(action_values, tie_break='random') == np.array([0, 0, 0, 1, 0])).all()

    # 1D with ties
    action_values = np.array([3, 4, -1, -1, 4, 0])
    assert greedy_policy(action_values) == 1
    assert greedy_policy(action_values, tie_break='last') == 4
    assert (greedy_policy(action_values, tie_break='random') == np.array([0, 0.5, 0, 0, 0.5, 0])).all()

    # 2D
    action_values = np.array([[-10, 3, -2, 11, 0, 0], [3, 4, -1, -1, 4, 0], [-1, 10, 10, 3, 10, 4], [4, 3, 2, 1, 0, -2]])
    assert (greedy_policy(action_values) == np.array([3, 1, 1, 0])).all()
    assert (greedy_policy(action_values, tie_break='last') == np.array([3, 4, 4, 0])).all()
    assert (greedy_policy(action_values, tie_break='random') == np.array([
    [0, 0, 0, 1, 0, 0], [0, 0.5, 0, 0, 0.5, 0], [0, 1/3, 1/3, 0, 1/3, 0], [1, 0, 0, 0, 0, 0]])).all()

def test_epsilon_greedy_policy():

    # 1D without ties
    action_values = np.array([-10, 3, -2, 11, 0])
    assert (epsilon_greedy_policy(action_values, eps=0) == greedy_policy(action_values, tie_break='random')).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2) == np.array([0.2/5, 0.2/5, 0.2/5, 0.2/5 + 0.8, 0.2/5])).all()
    assert (epsilon_greedy_policy(action_values, eps=1) == np.array([0.2, 0.2, 0.2, 0.2, 0.2])).all()

    # 1D with ties
    action_values = np.array([3, 4, -1, -1, 4, 0])
    assert (epsilon_greedy_policy(action_values, eps=0) == np.array([0, 1, 0, 0, 0, 0])).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2) == np.array([0.2/6, 0.2/6 + 0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6])).all()
    assert (epsilon_greedy_policy(action_values, eps=1) == np.array([1/6, 1/6,1/6, 1/6, 1/6, 1/6])).all()

    # 2D
    action_values = np.array([[-10, 3, -2, 11, 0, 0], [3, 4, -1, -1, 4, 0], [-1, 10, 10, 3, 10, 4], [4, 3, 2, 1, 0, -2]])
    assert (epsilon_greedy_policy(action_values, eps=0) == np.array([[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2) == np.array([
    [0.2/6, 0.2/6, 0.2/6, 0.2/6 + 0.8, 0.2/6, 0.2/6], [0.2/6, 0.2/6 + 0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6],
    [0.2/6, 0.2/6 + 0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6], [0.2/6 + 0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6, 0.2/6]])).all()
    assert (epsilon_greedy_policy(action_values, eps=1) == 1/6).all()

def test_ucb_policy():

    #1D
    action_values = np.array([0, 0, 0, 0, 0, 0])
    num_visits = np.array([3, 0, 7, 100, 1, 0])
    assert (ucb_policy(action_values, num_visits, 0) == greedy_policy(action_values)).all()
    assert ucb_policy(action_values, num_visits, 10) == 1
    assert ucb_policy(action_values, num_visits, 20, tie_break='last') == 5
    assert (ucb_policy(action_values, num_visits, 30, tie_break='random') == np.array([0, 0.5, 0, 0, 0, 0.5])).all()

    #2D
    action_values = np.array([[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]])
    num_visits = np.array([[3, 0, 7, 100, 1, 0], [3, 10, 7, 100, 20, 3]])
    assert (ucb_policy(action_values, num_visits, 0) == greedy_policy(action_values)).all()
    assert (ucb_policy(action_values, num_visits, 10) == np.array([1, 0])).all()
    assert (ucb_policy(action_values, num_visits, 20, tie_break='last') == np.array([5, 5])).all()
    assert (ucb_policy(action_values, num_visits, 30, tie_break='random') == np.array([[0, 0.5, 0, 0, 0, 0.5], [0.5, 0, 0, 0, 0, 0.5]])).all()

def test_softmax_policy():

    #1D
    action_preferences = np.array([-200, -100, 0, 100, 200])
    policy = softmax_policy(action_preferences)
    assert (policy[4] > policy[3]) and (policy[3] > policy[2]) and (policy[2] > policy[1]) and (policy[1] > policy[0])

    #2D
    action_preferences = np.array([[-200, -100, 0, 100, 200], [50, -25, -50, 0, 25]])
    policy = softmax_policy(action_preferences)
    assert (policy[0,4] > policy[0,3]) and (policy[0,3] > policy[0,2]) and (policy[0,2] > policy[0,1]) and (policy[0,1] > policy[0,0])
    assert (policy[1,0] > policy[1,4]) and (policy[1,4] > policy[1,3]) and (policy[1,3] > policy[1,1]) and (policy[1,1] > policy[1,2])
