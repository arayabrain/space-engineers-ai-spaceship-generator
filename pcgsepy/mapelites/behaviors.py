from typing import Tuple
import numpy as np

from ..common.vecs import Vec
from ..lsystem.solution import CandidateSolution


class BehaviorCharacterization:
    def __init__(self,
                 name: str,
                 func: callable,
                 bounds: Tuple[float, float]):
        self.name = name
        self.bounds = bounds
        self.f = func

    def __call__(self,
                 cs: CandidateSolution) -> float:
        return self.f(cs)


def mame(cs: CandidateSolution) -> float:
    volume = cs.content.as_array().shape
    largest_axis, medium_axis, _ = reversed(sorted(list(volume)))
    return largest_axis / medium_axis


def mami(cs: CandidateSolution) -> float:
    volume = cs.content.as_array().shape
    largest_axis, _, smallest_axis = reversed(sorted(list(volume)))
    return largest_axis / smallest_axis


def avg_ma(cs: CandidateSolution) -> float:
    volume = cs.content.as_array().shape
    largest_axis, medium_axis, smallest_axis = reversed(sorted(list(volume)))
    return ((largest_axis / medium_axis) + (largest_axis / smallest_axis)) / 2


def symmetry(cs: CandidateSolution):
    structure = cs.content
    blocks = structure._blocks.values()
    bts, bps = [], []
    center_pos = Vec.v3i(0, 0, 0)
    for i, block in enumerate(blocks):
        bts.append(block.block_type)
        bps.append(Vec.v3i(x=int(block.position.x / structure.grid_size),
                           y=int(block.position.y / structure.grid_size),
                           z=int(block.position.z / structure.grid_size)))
        if block.block_type == 'MyObjectBuilder_Cockpit_OpenCockpitLarge':
            center_pos = bps[-1]
    center_pos.x -= 1
    diff = Vec.v3i(x=-center_pos.x,
                   y=0,
                   z=-center_pos.z)
    for i in range(len(bps)):
        bps[i] = bps[i].sum(diff)
    err_x, err_z = 0, 0
    excl_x, excl_z = 0, 0
    for i, pos in enumerate(bps):
        # error along x
        if pos.x != 0:
            mirr = pos.sum(Vec.v3i(x=2*(-pos.x),
                                   y=0,
                                   z=0))
            try:
                other = bps.index(mirr)
                err_x += 1 if bts[i] != bts[other] else 0
            except ValueError:
                err_x += 1
        else:
            excl_x += 1
        # error along z
        if pos.z != 0:
            mirr = pos.sum(Vec.v3i(x=0,
                                   y=0,
                                   z=2*(-pos.z)))
            try:
                other = bps.index(mirr)
                err_z += 1 if bts[i] != bts[other] else 0
            except ValueError:
                err_z += 1
        else:
            excl_z += 1

    symm_x = 1 - ((err_x / 2) / (len(bps) - excl_x))
    symm_z = 1 - ((err_z / 2) / (len(bps) - excl_z))
    return max(symm_x, symm_z)