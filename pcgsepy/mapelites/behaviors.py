from typing import Any, Dict, Tuple
import numpy as np

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


def symmetry(cs: CandidateSolution):
    """Symmetry of the solution, expressed in `[0,1]`.

    Args:
        cs (CandidateSolution): The solution.

    Returns:
        _type_: The value of this behavior characterization.
    """
    structure = cs.content
    pivot_blocktype = 'MyObjectBuilder_Cockpit_OpenCockpitLarge'
    midpoint = [x for x in structure._blocks.values() if x.block_type == pivot_blocktype][0].position.scale(1 / structure.grid_size).to_veci()
    arr = structure.as_grid_array
    
    # along x
    x_shape = max(midpoint.x, arr.shape[0] - midpoint.x)
    upper = np.zeros((x_shape, arr.shape[1], arr.shape[2]))
    lower = np.zeros((x_shape, arr.shape[1], arr.shape[2]))
    upper[np.nonzero(np.flip(arr[midpoint.x:, :, :], 1))] = np.flip(arr[midpoint.x:, :, :], 1)[np.nonzero(np.flip(arr[midpoint.x:, :, :], 1))]
    lower[np.nonzero(arr[:midpoint.x - 1, :, :])] = arr[np.nonzero(arr[:midpoint.x - 1, :, :])]
    err_x = abs(np.sum(upper - lower))
    
    # along z
    z_shape = max(midpoint.z, arr.shape[2] - midpoint.z)
    upper = np.zeros((arr.shape[0], arr.shape[1], z_shape))
    lower = np.zeros((arr.shape[0], arr.shape[1], z_shape))
    tmp = np.flip(arr[:, :, midpoint.z:], 2)
    upper[np.nonzero(np.flip(arr[:, :, midpoint.z:], 2))] = np.flip(arr[:, :, midpoint.z:], 2)[np.nonzero(np.flip(arr[:, :, midpoint.z:], 2))]
    lower[np.nonzero(arr[:, :, :midpoint.z - 1])] = arr[np.nonzero(arr[:, :, :midpoint.z - 1])]
    err_z = abs(np.sum(upper - lower))
        
    return 1 - (min(err_x, err_z) / np.sum(arr))


behavior_funcs = {
    'mame': mame,
    'mami': mami,
    'avg_ma': avg_ma,
    'symmetry': symmetry
}
