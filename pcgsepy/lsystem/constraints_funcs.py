from typing import Any, Dict

import numpy as np
from pcgsepy.config import MAME_MEAN, MAME_STD, MAMI_MEAN, MAMI_STD
from pcgsepy.lsystem.solution import CandidateSolution


def components_constraint(cs: CandidateSolution,
                          extra_args: Dict[str, Any]) -> bool:
    req_tiles = extra_args['req_tiles']
    components_ok = True
    for c in req_tiles:
        components_ok &= c in cs.string
    return components_ok


def intersection_constraint(cs: CandidateSolution,
                            extra_args: Dict[str, Any]) -> bool:
    return cs.content.has_intersections


def symmetry_constraint(cs: CandidateSolution,
                        extra_args: Dict[str, Any]) -> bool:
    structure = cs.content.as_array
    is_symmetric = False
    for dim in range(3):
        is_symmetric |= np.array_equal(structure, np.flip(structure, axis=dim))
    return is_symmetric


def axis_constraint(cs: CandidateSolution,
                    extra_args: Dict[str, Any]) -> bool:
    volume = cs.content.as_array.shape
    largest_axis, medium_axis, smallest_axis = reversed(sorted(list(volume)))
    mame = largest_axis / medium_axis
    mami = largest_axis / smallest_axis
    sat = True
    sat &= MAME_MEAN - MAME_STD <= mame <= MAME_MEAN + MAME_STD
    sat &= MAMI_MEAN - MAMI_STD <= mami <= MAMI_MEAN + MAMI_STD
    return sat
