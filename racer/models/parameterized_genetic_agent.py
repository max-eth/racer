from sacred import Ingredient
from racer.car_racing_env import car_racing_env
from racer.models.genetic_agent import genetic
from racer.models.genetic_agent import GeneticAgent
from racer.methods.genetic_programming.parameterized import ParameterizedIndividual

parameterized_genetic = Ingredient(
    "parameterized_genetic", ingredients=[car_racing_env, genetic]
)


@parameterized_genetic.config
def genetic_config():
    individual_fname = "best_85.pkl"  # from experiment 413


class ParameterizedGeneticAgent(GeneticAgent):
    @parameterized_genetic.capture
    def __init__(self, individual_fname):
        parameterized_policy = ParameterizedIndividual.from_pickled_individual(
            individual_fname
        )
        super(ParameterizedGeneticAgent, self).__init__(
            policy_function=parameterized_policy
        )

    def get_flat_parameters(self):
        return self.policy_function.get_flat_parameters()

    def set_flat_parameters(self, flat_params):
        self.policy_function.set_flat_parameters(flat_params)
