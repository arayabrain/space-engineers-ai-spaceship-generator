from typing import Optional
import numpy as np

from ..lsystem.solution import CandidateSolution


class BehaviorCharacterization:
    def __init__(self,
                 name: str,
                 func: callable,
                 bounds: Optional[str] = None):
        self.name = name
        self.bounds = bounds
        self.f = func

    def __call__(self,
                 cs: CandidateSolution) -> float:
        return self.f(cs)


def mame(cs: CandidateSolution) -> float:
    volume = cs.content.as_array().shape
    largest_axis, medium_axis, smallest_axis = reversed(sorted(
        list(volume)))
    return largest_axis / medium_axis


def mami(cs: CandidateSolution) -> float:
    volume = cs.content.as_array().shape
    largest_axis, medium_axis, smallest_axis = reversed(sorted(
        list(volume)))
    return largest_axis / smallest_axis


def avg_ma(cs: CandidateSolution) -> float:
    volume = cs.content.as_array().shape
    largest_axis, medium_axis, smallest_axis = reversed(sorted(
        list(volume)))
    return ((largest_axis / medium_axis) + (largest_axis / smallest_axis)) / 2


def symmetry(cs: CandidateSolution):
    structure = cs.content.as_array()
    all_points = len(np.nonzero(structure)[0])
    # compute symmetry along x
    left_half = structure[:structure.shape[0] // 2, :, :]
    right_half = structure[structure.shape[0] // 2:, :, :]
    right_half = np.flip(right_half, axis=0)
    if left_half.shape[0] > right_half.shape[0]:
        left_half = left_half[:right_half.shape[0]]
    if right_half.shape[0] > left_half.shape[0]:
        right_half = right_half[:left_half.shape[0]]
    dx = len(np.nonzero(left_half - right_half)[0])
    symm_x = dx / all_points
    # compute symmetry along z
    lower_half = structure[:, :, :structure.shape[2] // 2]
    upper_half = structure[:, :, structure.shape[2] // 2:]
    upper_half = np.flip(upper_half, axis=2)
    if lower_half.shape[2] > upper_half.shape[2]:
        lower_half = lower_half[:, :, :upper_half.shape[2]]
    if upper_half.shape[2] > lower_half.shape[2]:
        upper_half = upper_half[:, :, :lower_half.shape[2]]
    dz = len(np.nonzero(lower_half - upper_half)[2])
    symm_z = dz / all_points
    return 1 - min(symm_x, symm_z)