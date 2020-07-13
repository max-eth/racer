from racer.car_racing_env import get_env


def eval(agent):
    env = get_env()
    return agent.evaluate(env)
