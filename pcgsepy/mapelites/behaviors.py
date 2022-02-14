from typing import Optional

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