from .traffic_time_env import ContextualTrafficTimeEnv, TrafficTimeEnv

def test_run():

    context_env = ContextualTrafficTimeEnv()
    context_env.reset()
    [context_env.step(context, k) for context in (0,1) for k in range(10)]

    env = TrafficTimeEnv()
    env.reset()
    [env.step(0, k) for k in range(10)]
