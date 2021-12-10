from enum import Enum, auto
from typing import Any, Callable, Dict


class ConstraintLevel(Enum):
    SOFT_CONSTRAINT = auto()
    HARD_CONSTRAINT = auto()


class ConstraintTime(Enum):
    DURING = auto()
    END = auto()


class ConstraintHandler:
    def __init__(self,
                 name: str,
                 level: ConstraintLevel,
                 when: ConstraintTime,
                 f: Callable[[str, Dict[str, Any]], bool],
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
