import numpy as np
from .generalized_policy_iteration import PolicyIteration, ValueIteration

def get_params():

    num_states = 25
    num_actions = 4
    initializer_params = {'init_method': 'zero'}
    terminal_states = (12)
    theta = 10e-4
    discount_factor = 1
    reward_dynamics = np.array([
    [-3, -3, -2, -3],
    [-3, -3, -2, -3],
    [-3, -3, -2, -3],
    [-3, -3, -2, -3],
    [-3, -3, -2, -3],
    [-3, -2, -1, -2],
    [-3, -2, -1, -2],
    [-3, -2, 0, -2],
    [-3, -2, -1, -2],
    [-3, -2, -1, -2],
    [-2, -1, -2, -1],
    [-2, 0, -2, -1],
    [0, 0, 0, 0],
    [-2, -1, -2, 0],
    [-2, -1, -2, -1],
    [-1, -2, -3, -2],
    [-1, -2, -3, -2],
    [0, -2, -3, -2],
    [-1, -2, -3, -2],
    [-1, -2, -3, -2],
    [-2, -3, -3, -3],
    [-2, -3, -3, -3],
    [-2, -3, -3, -3],
    [-2, -3, -3, -3],
    [-2, -3, -3, -3]
    ])

    next_state_dynamics = np.zeros((num_states, num_actions, num_states))
    for i in range(5):
        for j in range(5):
            s = 5*j + i
            if s == 12:
                next_state_dynamics[s, :, s] = 1
            else:
                if j > 0:
                    next_state_dynamics[s, 0, s-5] = 1
                else:
                    next_state_dynamics[s, 0, s] = 1
                if i < 4:
                    next_state_dynamics[s, 1, s+1] = 1
                else:
                    next_state_dynamics[s, 1, s] = 1
                if j < 4:
                    next_state_dynamics[s, 2, s+5] = 1
                else:
                    next_state_dynamics[s, 2, s] = 1
                if i > 0:
                    next_state_dynamics[s, 3, s-1] = 1
                else:
                    next_state_dynamics[s, 3, s] = 1


    return num_states, num_actions, initializer_params, terminal_states, theta, reward_dynamics, next_state_dynamics, discount_factor

def test_policy_evaluation():

    polit = PolicyIteration(*get_params())

    ans = np.array([-3, -3, -2, -3, -3, -1, -3, 0, -3, -1, -2, -2, 0, -2, -2, -3, -1, 0, -1, -3, -3, -3, -2, -3, -3])
    polit.reset()
    polit.policy = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
    ])
    polit.policy_evaluation(max_iterations=1)
    assert (polit.state_values.astype(int) == ans).all()

    ans = np.array([-8, -5, -2, -5, -8, -14, -8, 0, -8, -14, -13, -10, 0, -10, -13, -11, -11, 0, -11, -11, -8, -5, -2, -5, -8])
    polit.policy_evaluation(max_iterations=100)
    assert (polit.state_values.astype(int) == ans).all()

def test_run():

    ans = np.array([-4, -3, -2, -3, -4, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, -4, -3, -2, -3, -4])

    polit = PolicyIteration(*get_params())
    polit.reset()
    polit.run(max_iterations=100, max_eval_iterations=100)
    assert (polit.state_values.astype(int) == ans).all()

    valit = ValueIteration(*get_params())
    valit.reset()
    valit.run(max_iterations=100)
    assert (valit.state_values.astype(int) == ans).all()
