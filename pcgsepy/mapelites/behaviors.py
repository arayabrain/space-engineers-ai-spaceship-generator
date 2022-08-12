from typing import Any, Dict, Tuple

from pcgsepy.common.vecs import Vec
from pcgsepy.lsystem.solution import CandidateSolution


class BehaviorCharacterization:
    def __init__(self,
                 name: str,
                 func: callable,
                 bounds: Tuple[float, float]):
        """Create a behavior characterization object.

        Args:
            name (str): The name.
            func (callable): The function to compute.
            bounds (Tuple[float, float]): The upper and lower bounds.
        """
        self.name = name
        self.bounds = bounds
        self.f = func

    def __call__(self,
                 cs: CandidateSolution) -> float:
        return self.f(cs)

    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'bounds': list(self.bounds),
            'f': self.f.__name__
        }

    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'BehaviorCharacterization':
        return BehaviorCharacterization(name=my_args['name'],
                                        func=behavior_funcs[my_args['f']],
                                        bounds=tuple(my_args['bounds']))


def mame(cs: CandidateSolution) -> float:
    """Major axis over Medium axis.

    Args:
        cs (CandidateSolution): The solution.

    Returns:
        float: The value of this behavior characterization.
    """
    largest_axis, medium_axis, _ = reversed(sorted(list(cs.content.as_grid_array.shape)))
    return largest_axis / medium_axis


def mami(cs: CandidateSolution) -> float:
    """Major axis over Minimum axis.

    Args:
        cs (CandidateSolution): The solution.

    Returns:
        float: The value of this behavior characterization.
    """
    largest_axis, _, smallest_axis = reversed(sorted(list(cs.content.as_grid_array.shape)))
    return largest_axis / smallest_axis


def avg_ma(cs: CandidateSolution) -> float:
    """The average axis proportions.

    Args:
        cs (CandidateSolution): The solution.

    Returns:
        float: The value of this behavior characterization.
    """
    largest_axis, medium_axis, smallest_axis = reversed(sorted(list(cs.content.as_grid_array.shape)))
    return ((largest_axis / medium_axis) + (largest_axis / smallest_axis)) / 2


# TODO: This function should be optimized if possible.
def symmetry(cs: CandidateSolution):
    """Symmetry of the solution, expressed in `[0,1]`.

    Args:
        cs (CandidateSolution): The solution.

    Returns:
        _type_: The value of this behavior characterization.
    """
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
            mirr = pos.sum(Vec.v3i(x=2 * (-pos.x),
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


behavior_funcs = {
    'mame': mame,
    'mami': mami,
    'avg_ma': avg_ma,
    'symmetry': symmetry
}
