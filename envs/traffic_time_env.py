import numpy as np

class ContextualTrafficTimeEnv:
    """
    This environment simulates the time (in minutes) spent in traffic given the route chosen and the weather.

    The time in traffic for each route and weather is assumed to follow a normal distribution N(mean, std).
    There are 10 possible routes to be taken and 2 different weathers. Each time step is assumed to be a different day.

    Contexts:

        0: rainy day (20% chance)
        1: sunny day (80% chance)

    Actions: k for k in the range [0, 9] is the route number chosen

    Reward: negative of the time spent in traffic in route k in context s
    """

    IS_EPISODIC = False

    AVGS_NO_RAIN = [-23, -27, -18, -31, -40, -15, -18, -25, -22, -20]
    STDEVS_NO_RAIN = [6, 2, 7, 10, 7, 5, 5, 3, 8, 6]

    AVGS_RAIN = [-31, -37, -25, -40, -60, -27, -20, -32, -35, -29]
    STDEVS_RAIN = [6, 2, 7, 10, 7, 5, 5, 3, 8, 6]

    P_RAIN = 0.2

    def reset(self):
        if np.random.uniform() > self.P_RAIN:
            return 0, None
        return 1, None

    def step(self, context, route):

        if context == 0:
            time = np.random.normal(self.AVGS_NO_RAIN[route], self.STDEVS_NO_RAIN[route])
        elif context == 1:
            time = np.random.normal(self.AVGS_RAIN[route], self.STDEVS_RAIN[route])
        else:
            raise ValueError(f"The context must be either 0 or 1.")

        next_context = 0
        if np.random.uniform() < self.P_RAIN:
            next_context = 1

        return next_context, time, False, None

class TrafficTimeEnv(ContextualTrafficTimeEnv):
    """
    This environment is the same as ContextualTrafficTimeEnv, but here we don't condition on the weather.
    """

    def reset(self):
        return 0, None

    def step(self, context, route):

        assert context == 0, "Context can only take value 0"
        if np.random.uniform() < self.P_RAIN:
            time = np.random.normal(self.AVGS_RAIN[route], self.STDEVS_RAIN[route])
        else:
            time = np.random.normal(self.AVGS_NO_RAIN[route], self.STDEVS_NO_RAIN[route])

        return 0, time, False, None
