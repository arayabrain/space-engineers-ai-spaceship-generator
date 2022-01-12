from typing import Any, Dict
import numpy as np

from pcgsepy.common.vecs import Orientation, Vec
from pcgsepy.config import USE_BBOX, BBOX_X, BBOX_Y, BBOX_Z
from pcgsepy.lsystem.structure_maker import LLStructureMaker
from pcgsepy.structure import Structure


def dimension_reaching_fitness(axiom: str, extra_args: Dict[str, Any]) -> float:
    base_position, orientation_forward, orientation_up = Vec.v3i(
        0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                 position=base_position).fill_structure(
                                     structure=structure,
                                     axiom=axiom,
                                     additional_args={})
    structure.update(origin=base_position,
                     orientation_forward=orientation_forward,
                     orientation_up=orientation_up)

    x, y, z = structure._max_dims
    f = BBOX_X - abs(BBOX_X - x)
    f += BBOX_Y - abs(BBOX_Y - y)
    f += BBOX_Z - abs(BBOX_Z - z)
    return f


def box_filling_fitness(axiom: str, extra_args: Dict[str, Any]) -> float:
    base_position, orientation_forward, orientation_up = Vec.v3i(
        0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
    structure = Structure(origin=base_position,
                          orientation_forward=orientation_forward,
                          orientation_up=orientation_up)
    structure = LLStructureMaker(atoms_alphabet=extra_args['alphabet'],
                                 position=base_position).fill_structure(
                                     structure=structure,
                                     axiom=axiom,
                                     additional_args={})
    structure.update(origin=base_position,
                     orientation_forward=orientation_forward,
                     orientation_up=orientation_up)

    # filled boxed volume as percentage
    ps, _, _ = np.nonzero(structure.as_array()[:int(BBOX_X), :int(BBOX_Y), :int(BBOX_Z)])
    actual_v = len(ps)
    ideal_v = BBOX_X * BBOX_Y * BBOX_Z
    filled_v = actual_v / ideal_v
    f = 1 - abs(1 - filled_v)

    return 100*f