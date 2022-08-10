import math
import pickle
from typing import Any, Callable, Dict, Tuple

import numpy as np
from pcgsepy.config import BBOX_X, BBOX_Y, BBOX_Z
from pcgsepy.lsystem.solution import CandidateSolution
from scipy.stats import gaussian_kde

# load pickled estimators
with open('./estimators/futo.pkl', 'rb') as f:
    futo_es: gaussian_kde = pickle.load(f)
with open('./estimators/tovo.pkl', 'rb') as f:
    tovo_es: gaussian_kde = pickle.load(f)
with open('./estimators/mame.pkl', 'rb') as f:
    mame_es: gaussian_kde = pickle.load(f)
with open('./estimators/mami.pkl', 'rb') as f:
    mami_es: gaussian_kde = pickle.load(f)
# Compute max values for estimators to normalize fitnesses
# values are chosen upon inspection.
x = np.linspace(0, 0.5, int(0.5 / 0.005))
futo_max = float(np.max(futo_es.evaluate(x)))
x = np.linspace(0, 1, int(1 / 0.005))
tovo_max = float(np.max(tovo_es.evaluate(x)))
x = np.linspace(0, 6, int(6 / 0.005))
mame_max = float(np.max(mame_es.evaluate(x)))
x = np.linspace(0, 10, int(10 / 0.005))
mami_max = float(np.max(mami_es.evaluate(x)))


def bounding_box_fitness(cs: CandidateSolution,
                         extra_args: Dict[str, Any]) -> float:
    """Measure how close the structure fits in the bounding box.
    Penalizes in both ways.
    Normalized in [0,1].

    Args:
        cs (CandidateSolution): The candidate solution.
        extra_args (Dict[str, Any]): Extra arguments.

    Returns:
        float: The fitness value.
    """
    x, y, z = cs.content.as_array().shape
    f = np.clip((BBOX_X - abs(BBOX_X - x)) / BBOX_X, 0, 1)
    f += np.clip((BBOX_Y - abs(BBOX_Y - y)) / BBOX_Y, 0, 1)
    f += np.clip((BBOX_Z - abs(BBOX_Z - z)) / BBOX_Z, 0, 1)
    return f[0] / 3


def box_filling_fitness(cs: CandidateSolution,
                        extra_args: Dict[str, Any]) -> float:
    """Measures how much of the total volume is filled with blocks.
    Normalized in [0,1].

    Args:
        cs (CandidateSolution): The candidate solution.
        extra_args (Dict[str, Any]): Extra arguments.

    Returns:
        float: The fitness value.
    """
    return tovo_es.evaluate(sum([b.volume for b in cs.content._blocks.values()]) / math.prod(cs.content.as_array().shape))[0] / tovo_max


def func_blocks_fitness(cs: CandidateSolution,
                        extra_args: Dict[str, Any]) -> float:
    """Measures how much of the total blocks is functional blocks.
    Normalized in [0,1].

    Args:
        cs (CandidateSolution): The candidate solution.
        extra_args (Dict[str, Any]): Extra arguments.

    Returns:
        float: The fitness value.
    """
    fu, to = 0., 0.
    for b in cs.content._blocks.values():
        fu += b.volume if not b.block_type.startswith(
            'MyObjectBuilder_CubeBlock_') else 0
        to += b.volume
    return futo_es.evaluate(fu / to)[0] / futo_max


def mame_fitness(cs: CandidateSolution,
                 extra_args: Dict[str, Any]) -> float:
    """Measures the proportions of the largest and medium axis.
    Normalized in [0,1].

    Args:
        cs (CandidateSolution): The candidate solution.
        extra_args (Dict[str, Any]): Extra arguments.

    Returns:
        float: The fitness value.
    """
    largest_axis, medium_axis, _ = reversed(
        sorted(list(cs.content.as_array().shape)))
    return mame_es.evaluate(largest_axis / medium_axis)[0] / mame_max


def mami_fitness(cs: CandidateSolution,
                 extra_args: Dict[str, Any]) -> float:
    """Measures the proportions of the largest and smallest axis.
    Normalized in [0,1]

    Args:
        cs (CandidateSolution): The candidate solution.
        extra_args (Dict[str, Any]): Extra arguments.

    Returns:
        float: The fitness value.
    """
    largest_axis, _, smallest_axis = reversed(
        sorted(list(cs.content.as_array().shape)))
    return mami_es.evaluate(largest_axis / smallest_axis)[0] / mami_max


fitness_functions = {
    'bounding_box_fitness': bounding_box_fitness,
    'box_filling_fitness': box_filling_fitness,
    'func_blocks_fitness': func_blocks_fitness,
    'mame_fitness': mame_fitness,
    'mami_fitness': mami_fitness
}


class Fitness:
    def __init__(self,
                 name: str,
                 f: Callable[[CandidateSolution, Dict[str, Any]], float],
                 bounds: Tuple[float, float],
                 weight: float = 1.0):
        self.name = name
        self.f = f
        self.bounds = bounds
        self.weight = weight

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return f'Fitness {self.name} (in {self.bounds})'

    def __call__(self,
                 cs: CandidateSolution,
                 extra_args: Dict[str, Any]) -> float:
        return self.f(cs=cs,
                      extra_args=extra_args)

    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'f': self.f.__name__,
            'bounds': list(self.bounds),
            'weight': self.weight
        }

    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'Fitness':
        return Fitness(name=my_args['name'],
                       f=fitness_functions[my_args['f']],
                       bounds=tuple(my_args['bounds']),
                       weight=my_args['weight'])
