import multineat

from revolve2.core.modular_robot import Body

from .._genotype import Genotype
from .._random_v1 import random_v1 as base_random_v1
from .brain_ann import BrainANN


def random_v1(
    innov_db: multineat.InnovationDatabase,
    rng: multineat.RNG,
    multineat_params: multineat.Parameters,
    output_activation_func: multineat.ActivationFunction,
    num_initial_mutations: int,
    n_env_conditions: int,
    plastic_brain: int,
) -> Genotype:
    assert multineat_params.MutateOutputActivationFunction == False
    # other activation functions could work too, but this has been tested.
    # if you want another one, make sure it's output is between -1 and 1.
    #assert output_activation_func == multineat.ActivationFunction.SIGNED_SINE

     # using more than 8 inpurs gives me segmentation fault! why?
    return base_random_v1(
        innov_db,
        rng,
        multineat_params,
        output_activation_func,
        8, #spider (8 joints)
        8,
        num_initial_mutations,
    )


def develop_v1(genotype: Genotype, body: Body, env_condition: int,
               n_env_conditions: int, plastic_brain: int) -> BrainANN:
    return BrainANN(genotype.genotype)
