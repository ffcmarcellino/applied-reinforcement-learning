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
    mask = np.array([True, True, True, False, True])
    assert greedy_policy(action_values) == greedy_policy(action_values, tie_break='last') == 3
    assert greedy_policy(action_values, mask=mask) == greedy_policy(action_values, tie_break='last', mask=mask) == 1
    assert (greedy_policy(action_values, tie_break='random') == np.array([0, 0, 0, 1, 0])).all()
    assert (greedy_policy(action_values, tie_break='random', mask=mask) == np.array([0, 1, 0, 0, 0])).all()


    # 1D with ties
    action_values = np.array([3, 4, -1, -1, 4, 0])
    mask = np.array([True, False, True, True, True, True])
    assert greedy_policy(action_values) == 1
    assert greedy_policy(action_values, mask=mask) == greedy_policy(action_values, tie_break='last') == greedy_policy(action_values, tie_break='last', mask=mask) == 4
    assert (greedy_policy(action_values, tie_break='random') == np.array([0, 0.5, 0, 0, 0.5, 0])).all()
    assert (greedy_policy(action_values, tie_break='random', mask=mask) == np.array([0, 0, 0, 0, 1, 0])).all()

    # 2D
    action_values = np.array([[-10, 3, -2, 11, 0, 0], [3, 4, -1, -1, 4, 0], [-1, 10, 10, 3, 10, 4], [4, 3, 2, 1, 0, -2]])
    mask = np.array([[True, True, True, False, True, True], [True, True, True, True, False, True], [True, False, True, True, True, True], [False, False, True, True, True, True]])
    assert (greedy_policy(action_values) == np.array([3, 1, 1, 0])).all()
    assert (greedy_policy(action_values, mask=mask) == np.array([1, 1, 2, 2])).all()
    assert (greedy_policy(action_values, tie_break='last') == np.array([3, 4, 4, 0])).all()
    assert (greedy_policy(action_values, tie_break='last', mask=mask) == np.array([1, 1, 4, 2])).all()
    assert (greedy_policy(action_values, tie_break='random') == np.array([
    [0, 0, 0, 1, 0, 0], [0, 0.5, 0, 0, 0.5, 0], [0, 1/3, 1/3, 0, 1/3, 0], [1, 0, 0, 0, 0, 0]])).all()
    assert (greedy_policy(action_values, tie_break='random', mask=mask) == np.array([
    [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1/2, 0, 1/2, 0], [0, 0, 1, 0, 0, 0]])).all()

def test_epsilon_greedy_policy():

    # 1D without ties
    action_values = np.array([-10, 3, -2, 11, 0])
    mask = np.array([True, False, True, False, True])
    assert (epsilon_greedy_policy(action_values, eps=0) == greedy_policy(action_values, tie_break='random')).all()
    assert (epsilon_greedy_policy(action_values, eps=0, mask=mask) == greedy_policy(action_values, tie_break='random', mask=mask)).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2) == np.array([0.2/5, 0.2/5, 0.2/5, 0.2/5 + 0.8, 0.2/5])).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2, mask=mask) == np.array([0.2/3, 0, 0.2/3, 0, 0.8 + 0.2/3])).all()
    assert (epsilon_greedy_policy(action_values, eps=1) == 0.2).all()
    assert (epsilon_greedy_policy(action_values, eps=1, mask=mask) == np.array([1/3, 0, 1/3, 0, 1/3])).all()

    # 1D with ties
    action_values = np.array([3, 4, -1, -1, 4, 0])
    mask = np.array([True, False, True, True, True, False])
    assert (epsilon_greedy_policy(action_values, eps=0) == np.array([0, 1, 0, 0, 0, 0])).all()
    assert (epsilon_greedy_policy(action_values, eps=0, mask=mask) == np.array([0, 0, 0, 0, 1, 0])).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2) == np.array([0.2/6, 0.2/6 + 0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6])).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2, mask=mask) == np.array([0.2/4, 0, 0.2/4, 0.2/4, 0.8 + 0.2/4, 0])).all()
    assert (epsilon_greedy_policy(action_values, eps=1) == 1/6).all()
    assert (epsilon_greedy_policy(action_values, eps=1, mask=mask) == np.array([1/4, 0,1/4, 1/4, 1/4, 0])).all()

    # 2D
    action_values = np.array([[-10, 3, -2, 11, 0, 0], [3, 4, -1, -1, 4, 0], [-1, 10, 10, 3, 10, 4], [4, 3, 2, 1, 0, -2]])
    mask = np.array([[True, True, True, False, True, True], [True, True, True, True, False, True], [True, False, True, True, True, True], [False, False, True, True, True, True]])
    assert (epsilon_greedy_policy(action_values, eps=0) == np.array([[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])).all()
    assert (epsilon_greedy_policy(action_values, eps=0, mask=mask) == np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]])).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2) == np.array([
    [0.2/6, 0.2/6, 0.2/6, 0.2/6 + 0.8, 0.2/6, 0.2/6], [0.2/6, 0.2/6 + 0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6],
    [0.2/6, 0.2/6 + 0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6], [0.2/6 + 0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6, 0.2/6]])).all()
    assert (epsilon_greedy_policy(action_values, eps=0.2, mask=mask) == np.array([
    [0.2/5, 0.8 + 0.2/5, 0.2/5, 0, 0.2/5, 0.2/5], [0.2/5, 0.2/5 + 0.8, 0.2/5, 0.2/5, 0, 0.2/5],
    [0.2/5, 0, 0.8 + 0.2/5, 0.2/5, 0.2/5, 0.2/5], [0, 0, 0.8 + 0.2/4, 0.2/4, 0.2/4, 0.2/4]])).all()
    assert (epsilon_greedy_policy(action_values, eps=1) == 1/6).all()
    assert (epsilon_greedy_policy(action_values, eps=1, mask=mask) == np.array([
    [1/5, 1/5, 1/5, 0, 1/5, 1/5], [1/5, 1/5, 1/5, 1/5, 0, 1/5], [1/5, 0, 1/5, 1/5, 1/5, 1/5], [0, 0, 1/4, 1/4, 1/4, 1/4]])).all()

def test_ucb_policy():

    #1D
    action_values = np.array([0, 0, 0, 0, 0, 0])
    num_visits = np.array([3, 0, 7, 100, 1, 0])
    mask = np.array([True, False, True, True, True, False])
    assert (ucb_policy(action_values, num_visits, 0) == greedy_policy(action_values)).all()
    assert ucb_policy(action_values, num_visits, 0, mask=mask) == greedy_policy(action_values, mask=mask)
    assert ucb_policy(action_values, num_visits, 10) == 1
    assert ucb_policy(action_values, num_visits, 20, tie_break='last') == 5
    assert ucb_policy(action_values, num_visits, 10, mask=mask) == ucb_policy(action_values, num_visits, 20, tie_break='last', mask=mask) == 4
    assert (ucb_policy(action_values, num_visits, 30, tie_break='random') == np.array([0, 0.5, 0, 0, 0, 0.5])).all()
    assert (ucb_policy(action_values, num_visits, 30, tie_break='random', mask=mask) == np.array([0, 0, 0, 0, 1, 0])).all()

    #2D
    action_values = np.array([[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]])
    num_visits = np.array([[3, 0, 7, 100, 1, 0], [3, 10, 7, 100, 20, 3]])
    mask = np.array([[True, False, True, True, True, False], [False, True, True, True, True, False]])
    assert (ucb_policy(action_values, num_visits, 0) == greedy_policy(action_values)).all()
    assert (ucb_policy(action_values, num_visits, 0, mask=mask) == greedy_policy(action_values, mask=mask)).all()
    assert (ucb_policy(action_values, num_visits, 10) == np.array([1, 0])).all()
    assert (ucb_policy(action_values, num_visits, 20, tie_break='last') == np.array([5, 5])).all()
    assert (ucb_policy(action_values, num_visits, 10, mask=mask) == np.array([4, 2])).all()
    assert (ucb_policy(action_values, num_visits, 20, tie_break='last', mask=mask) == np.array([4, 2])).all()
    assert (ucb_policy(action_values, num_visits, 30, tie_break='random') == np.array([[0, 0.5, 0, 0, 0, 0.5], [0.5, 0, 0, 0, 0, 0.5]])).all()
    assert (ucb_policy(action_values, num_visits, 30, tie_break='random', mask=mask) == np.array([[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]])).all()

def test_softmax_policy():

    #1D
    action_preferences = np.array([-200, -100, 0, 100, 200])
    policy = softmax_policy(action_preferences)
    assert (policy > 0).all()
    assert (policy[4] > policy[3]) and (policy[3] > policy[2]) and (policy[2] > policy[1]) and (policy[1] > policy[0])
    mask = np.array([True, True, True, False, True])
    mask_policy = softmax_policy(action_preferences, mask=mask)
    np.testing.assert_allclose(policy[3], 0, atol=1e-8)
    assert (policy[[0,1,2,4]] > 0).all()
    assert (policy[4] > policy[2]) and (policy[2] > policy[1]) and (policy[1] > policy[0])
    

    #2D
    action_preferences = np.array([[-200, -100, 0, 100, 200], [50, -25, -50, 0, 25]])
    policy = softmax_policy(action_preferences)
    assert (policy > 0).all()
    assert (policy[0,4] > policy[0,3]) and (policy[0,3] > policy[0,2]) and (policy[0,2] > policy[0,1]) and (policy[0,1] > policy[0,0])
    assert (policy[1,0] > policy[1,4]) and (policy[1,4] > policy[1,3]) and (policy[1,3] > policy[1,1]) and (policy[1,1] > policy[1,2])
    mask = np.array([[True, True, True, False, True], [True, True, True, True, False]])
    assert (policy[[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 4, 0, 1, 2, 3]] > 0).all()
    np.testing.assert_allclose(policy[[0, 1], [3, 4]], 0, atol=1e-8)
    assert (policy[0,4] > policy[0,2]) and (policy[0,2] > policy[0,1]) and (policy[0,1] > policy[0,0])
    assert (policy[1,0] > policy[1,3]) and (policy[1,3] > policy[1,1]) and (policy[1,1] > policy[1,2])
    