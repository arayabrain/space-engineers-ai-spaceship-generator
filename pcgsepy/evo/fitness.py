from typing import Any, Dict
from pcgsepy.common.vecs import Orientation, Vec

from pcgsepy.config import USE_BBOX, BBOX_X, BBOX_Y, BBOX_Z
from pcgsepy.lsystem.structure_maker import LLStructureMaker
from pcgsepy.structure import Structure


def compute_fitness(axiom: str, extra_args: Dict[str, Any]) -> float:
    if not USE_BBOX:
        return 0.
    else:
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