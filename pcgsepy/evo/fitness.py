from typing import Any, Dict
from scipy.stats import norm
import numpy as np

from pcgsepy.common.vecs import Orientation, Vec
from pcgsepy.config import USE_BBOX, BBOX_X, BBOX_Y, BBOX_Z
from pcgsepy.config import TOVO_MEAN, TOVO_STD, FUTO_MEAN, FUTO_STD
from pcgsepy.lsystem.structure_maker import LLStructureMaker
from pcgsepy.structure import Structure


def bounding_box_fitness(axiom: str, extra_args: Dict[str, Any]) -> float:
    """
    Measure how close the structure fits in the bounding box.
    Penalizes in both ways.
    Normalized in [0,1].
    """
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

    x, y, z = structure.as_array().shape
    f = BBOX_X - abs(BBOX_X - x)
    f += BBOX_Y - abs(BBOX_Y - y)
    f += BBOX_Z - abs(BBOX_Z - z)
    return f / (BBOX_X + BBOX_Y + BBOX_Z)


def box_filling_fitness(axiom: str, extra_args: Dict[str, Any]) -> float:
    """
    Measures how much of the total volume is filled with blocks.
    Uses TOVO_MEAN and TOVO_STD.
    Normalized in [0,1]
    """
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

    to = sum([b.volume for b in structure.get_all_blocks(to_place=False)])
    vo = structure.as_array().shape
    vo = vo[0] * vo[1] * vo[2]
    tovo = to / vo
    return norm(tovo, TOVO_MEAN, TOVO_STD)


def func_blocks_fitness(axiom: str, extra_args: Dict[str, Any]) -> float:
    """
    Measures how much of the total blocks is functional blocks.
    Uses FUTO_MEAN and FUTO_STD.
    Normalized in [0,1]
    """
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
        b.block_type.startswith('MyObjectBuilder_CubeBlock_')])

    fu, to = 0., 0.
    for b in structure.get_all_blocks(to_place=False):
        if not b.block_type.startswith('MyObjectBuilder_CubeBlock_'):
            fu += b.volume
        to += b.volume
    futo = fu / to
    return norm(futo, FUTO_MEAN, FUTO_STD)