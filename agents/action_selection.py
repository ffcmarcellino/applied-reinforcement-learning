import numpy as np

""" Policy Action Selecion """

def policy_action_selection(policy: np.array):
    """Selects an action according to a specific policy.

    Args:
        numpy.array (num_actions,) 'policy': array of probabilities for each actions, where each index represents an action

    Returns:
        The action index to be taken by following the policy
    """

    covered_actions = np.where(policy > 0)[0]
    action_probs = policy[covered_actions]
    return covered_actions[0] if len(covered_actions)==1 else np.random.choice(covered_actions, p=action_probs)

""" Greedy Policy """

TIE_BREAK = (
    'first', # picks the first tied value
    'last', # picks the last tied value
    'random', # randomly picks one of the tied values with same probabilities
    )

def greedy_policy(action_values: np.array, tie_break='first'):
    """Computes a greedy policy with respect to the action-values

    Args:
        numpy.array (num_states, num_actions) or (num_actions,) 'action_values': array of action values
        str 'tie_break': method to be used to break ties of multiple greedy actions

    Returns:
        The action index to be taken or action probabilities (for each state), depending on tie_break
    """

    assert tie_break in TIE_BREAK, f"{tie_break} is not a valid value for tie_break argument."

    axis = None if len(action_values.shape) == 1 else 1
    num_actions = len(action_values) if axis is None else action_values.shape[1]

    if tie_break == 'first':
        return action_values.argmax(axis=axis)

    if tie_break == 'last':
        return num_actions - 1 - np.flip(action_values, axis=axis).argmax(axis=axis)

    if tie_break == 'random':
        max_values = np.expand_dims(action_values.max(axis=axis), -1)
        action_probs = np.equal(action_values, max_values)*1
        action_probs = action_probs/np.expand_dims(action_probs.sum(axis=axis), -1)
        return action_probs

""" Greedy Policy """

def epsilon_greedy_policy(action_values: np.array, eps: float):
    """Computes an epsilon-greedy policy with respect to the action values

    Args:
        numpy.array (num_states, num_actions) or (num_actions,) 'action_values': array of action values
        float 'eps': exploration parameter epsilon

    Returns:
        Probability of each action (for each state)
    """

    action_probs = np.zeros(action_values.shape)
    greedy_actions = greedy_policy(action_values)
    if len(action_values.shape) == 2:
        action_probs[np.arange(len(greedy_actions)), greedy_actions] = 1-eps
        action_probs += eps/action_values.shape[1]
    else:
        action_probs[greedy_actions] = 1-eps
        action_probs += eps/len(action_values)

    return action_probs

""" Upper Confidence Bound Policy """

def ucb_policy(action_values: np.array, num_visits: np.array, c: int, tie_break='first'):
    """Computes greedy policy with respect to action values adjusted by upper confidence bound method

    Args:
        numpy.array (num_states, num_actions) or (num_actions,) 'action_values': array of action-values, where each index represents an action
        numpy.array (num_states, num_actions) or (num_actions,) 'num_visits': number of visits that the agent has taken for each action, where each index represents an action
        int 'c': degree of exploration
        str 'tie_break': method to be used to break ties of multiple greedy actions

    Returns:
        The action index to be taken or action probabilities, depending on out_probs
    """
    axis = None if len(num_visits.shape) == 1 else 1
    num_visits_aux = num_visits + 1
    t = np.expand_dims(num_visits_aux.sum(axis=axis), -1)
    ucb_values = action_values + c*np.sqrt(np.log(t)/num_visits_aux)
    return greedy_policy(ucb_values, tie_break)

""" Softmax Policy """

def softmax_policy(action_preferences: np.array):
    """Computes a softmax policy with respect to action preferences.

    Args:
        numpy.array (num_states, num_actions) or (num_actions,) 'action_preferences': array of action preferences

    Returns:
        Probability of each action (for each state)
    """
    axis = None if len(action_preferences.shape) == 1 else 1
    exp_action_preferences = np.exp(action_preferences)
    action_probs = exp_action_preferences/np.expand_dims(exp_action_preferences.sum(axis=axis), -1)
    return action_probs
