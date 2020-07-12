from sacred import Experiment

from racer.car_racing_env import car_racing_env, get_env
from racer.utils import setup_sacred_experiment, flatten_parameters, build_parameters
from racer.models.simple_nn import NNAgent, simple_nn
from scipy.optimize import minimize

ex = Experiment("scipy_nelder_mead", ingredients=[car_racing_env, simple_nn],)
setup_sacred_experiment(ex)


@ex.automain
def run():

    real_env = get_env()

    agent = NNAgent()

    shapes = [p.shape for p in agent.parameters()]

    def f(params):
        real_env.reset(regen_track=False)
        agent.set_parameters(build_parameters(shapes, params))
        return agent.evaluate(real_env, visible=False)

    minimize(
        f,
        flatten_parameters(agent.parameters()),
        method="Nelder-Mead",
        options={"disp": True},
    )
