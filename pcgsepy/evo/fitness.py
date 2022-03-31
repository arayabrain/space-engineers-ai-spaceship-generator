import pickle
from typing import Any, Callable, Dict, Tuple

import numpy as np

from ..common.vecs import Orientation, Vec
from ..config import BBOX_X, BBOX_Y, BBOX_Z
from ..lsystem.solution import CandidateSolution
from ..lsystem.structure_maker import LLStructureMaker
from ..structure import Structure


class Fitness:
    def __init__(self,
                 name: str,
                 f: Callable[[CandidateSolution, Dict[str, Any]], float],
                 bounds: Tuple[int, int],
                 weight: float = 1.0) -> None:
        self.name = name
        self.f = f
        self.bounds = bounds
        self.weight = weight

    def __repr__(self) -> str:
        return f'Fitness {self.name} (in {self.bounds})'

    def __str__(self) -> str:
        return self.__repr__()

    def __call__(self,
                 cs: CandidateSolution,
                 extra_args: Dict[str, Any]) -> float:
        return self.f(cs=cs,
                      extra_args=extra_args)


# load pickled estimators
with open('./estimators/futo.pkl', 'rb') as f:
    futo_es = pickle.load(f)
with open('./estimators/tovo.pkl', 'rb') as f:
    tovo_es = pickle.load(f)
with open('./estimators/mame.pkl', 'rb') as f:
    mame_es = pickle.load(f)
with open('./estimators/mami.pkl', 'rb') as f:
    mami_es = pickle.load(f)
# Compute max values for estimators to normalize fitnesses
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
    """
    Measure how close the structure fits in the bounding box.
    Penalizes in both ways.
    Normalized in [0,1].
    """
    if cs._content is None:
        base_position, orientation_forward, orientation_up = Vec.v3i(
            0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
        structure = Structure(origin=base_position,
                              orientation_forward=orientation_forward,
                              orientation_up=orientation_up)
        structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                     position=base_position).fill_structure(structure=structure,
                                                                            string=cs.ll_string,
                                                                            additional_args={})
        structure.update(origin=base_position,
                         orientation_forward=orientation_forward,
                         orientation_up=orientation_up)
        cs.set_content(content=structure)

    x, y, z = cs.content.as_array().shape
    f = np.clip((BBOX_X - abs(BBOX_X - x)) / BBOX_X, 0, 1)
    f += np.clip((BBOX_Y - abs(BBOX_Y - y)) / BBOX_Y, 0, 1)
    f += np.clip((BBOX_Z - abs(BBOX_Z - z)) / BBOX_Z, 0, 1)
    return f[0] / 3


def box_filling_fitness(cs: CandidateSolution,
                        extra_args: Dict[str, Any]) -> float:
    """
    Measures how much of the total volume is filled with blocks.
    Uses TOVO_MEAN and TOVO_STD.
    Normalized in [0,1]
    """
    if cs._content is None:
        base_position, orientation_forward, orientation_up = Vec.v3i(
            0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
        structure = Structure(origin=base_position,
                              orientation_forward=orientation_forward,
                              orientation_up=orientation_up)
        structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                     position=base_position).fill_structure(structure=structure,
                                                                            string=cs.ll_string,
                                                                            additional_args={})
        structure.update(origin=base_position,
                         orientation_forward=orientation_forward,
                         orientation_up=orientation_up)
        cs.set_content(content=structure)

    to = sum([b.volume for b in cs.content._blocks.values()])
    vo = cs.content.as_array().shape
    vo = vo[0] * vo[1] * vo[2]
    tovo = to / vo
    return tovo_es.evaluate(tovo)[0] / tovo_max


def func_blocks_fitness(cs: CandidateSolution,
                        extra_args: Dict[str, Any]) -> float:
    """
    Measures how much of the total blocks is functional blocks.
    Uses FUTO_MEAN and FUTO_STD.
    Normalized in [0,1]
    """
    if cs._content is None:
        base_position, orientation_forward, orientation_up = Vec.v3i(
            0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
        structure = Structure(origin=base_position,
                              orientation_forward=orientation_forward,
                              orientation_up=orientation_up)
        structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                     position=base_position).fill_structure(structure=structure,
                                                                            string=cs.ll_string,
                                                                            additional_args={})
        structure.update(origin=base_position,
                         orientation_forward=orientation_forward,
                         orientation_up=orientation_up)
        cs.set_content(content=structure)

    fu, to = 0., 0.
    for b in cs.content._blocks.values():
        if not b.block_type.startswith('MyObjectBuilder_CubeBlock_'):
            fu += b.volume
        to += b.volume
    futo = fu / to
    return futo_es.evaluate(futo)[0] / futo_max


def axis_fitness(cs: CandidateSolution,
                 extra_args: Dict[str, Any]) -> float:
    """
    Measures how much of the total blocks is functional blocks.
    Uses FUTO_MEAN and FUTO_STD.
    Normalized in [0,2]
    """
    if cs._content is None:
        base_position, orientation_forward, orientation_up = Vec.v3i(
            0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
        structure = Structure(origin=base_position,
                              orientation_forward=orientation_forward,
                              orientation_up=orientation_up)
        structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                     position=base_position).fill_structure(structure=structure,
                                                                            string=cs.ll_string,
                                                                            additional_args={})
        structure.update(origin=base_position,
                         orientation_forward=orientation_forward,
                         orientation_up=orientation_up)
        cs.set_content(content=structure)

    volume = cs.content.as_array().shape
    largest_axis, medium_axis, smallest_axis = reversed(sorted(list(volume)))
    mame = largest_axis / medium_axis
    mami = largest_axis / smallest_axis
    return (mame_es.evaluate(mame)[0] / mame_max) + (mami_es.evaluate(mami)[0] / mami_max)
