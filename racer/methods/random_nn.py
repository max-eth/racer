from sacred import Experiment

from racer.car_racing_env import car_racing_env, get_track_data, get_env
from racer.utils import setup_sacred_experiment
from racer.models.simple_nn import simple_nn, NNAgent

ex = Experiment("random_nn", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.automain
def main():
    real_env = get_env()

    agent = NNAgent()
    print("Running with real environment")
    print(agent.evaluate(real_env, True))
    real_env.reset(regen_track=False)

    print("Running with real environment again")
    print(agent.evaluate(real_env, True))
    real_env.viewer.close()

    print("Running with fake environment")
    fake_env = get_env(track_data=real_env.export())
    print(agent.evaluate(fake_env, True))
    fake_env.viewer.close()

    print("Running with fake environment again")
    fake_env = get_env(track_data=real_env.export())
    print(agent.evaluate(fake_env, False))

    print("Running with fake environment and viewer")
    fake_env = get_env(track_data=real_env.export(), render_view=True)
    print(agent.evaluate(fake_env, False))

    print("Running with fake environment after reset")
    fake_env.reset(regen_track=False)
    print(agent.evaluate(fake_env, False))
