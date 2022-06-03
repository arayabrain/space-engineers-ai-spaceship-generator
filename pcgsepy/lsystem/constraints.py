from enum import IntEnum, auto
from typing import Any, Callable, Dict

from pcgsepy.lsystem.constraints_funcs import *

from .solution import CandidateSolution


constraint_funcs = {
    'components_constraint': components_constraint,
    'intersection_constraint': intersection_constraint,
    'symmetry_constraint': symmetry_constraint,
    'axis_constraint': axis_constraint,
}


class ConstraintLevel(IntEnum):
    SOFT_CONSTRAINT = auto()
    HARD_CONSTRAINT = auto()


class ConstraintTime(IntEnum):
    DURING = auto()
    END = auto()


class ConstraintHandler:

    def __init__(self,
                 name: str,
                 level: ConstraintLevel,
                 when: ConstraintTime,
                 f: Callable[[CandidateSolution, Dict[str, Any]], bool],
                 extra_args: Dict[str, Any],
                 needs_ll: bool = False):
        self.name = name
        self.level = level
        self.when = when
        self.needs_ll = needs_ll
        self.constraint = f
        self.extra_args = extra_args

    def __repr__(self) -> str:
        return f'Constraint {self.name} ({self.level.name}) at {self.when.name}'

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash((self.name, self.level.value, self.when.value,
                     str(self.extra_args)))

    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'level': self.level.value,
            'when': self.when.value,
            'needs_ll': self.needs_ll,
            'constraint': self.constraint.__name__,
            'extra_args': self.extra_args
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'ConstraintHandler':
        return ConstraintHandler(name=my_args['name'],
                                 level=ConstraintLevel(my_args['level']),
                                 when=ConstraintTime(my_args['when']),
                                 f=constraint_funcs[my_args['constraint']],
                                 extra_args=my_args['extra_args'],
                                 needs_ll=my_args['needs_ll'])