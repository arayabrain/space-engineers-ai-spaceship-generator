from typing import Dict

from .actions import *
from .constraints import *
from .parser import LParser
from ..structure import *


orientations_ids = {
    "U": Orientation.UP,
    "D": Orientation.DOWN,
    "L": Orientation.LEFT,
    "R": Orientation.RIGHT,
    "F": Orientation.FORWARD,
    "B": Orientation.BACKWARD,
}

# Note: This can definitely be simplified, but for now it *works*
_rotate_orientations = {
    Orientation.UP: {
        Rotations.XcwY: Orientation.RIGHT,
        Rotations.XccwY: Orientation.LEFT,
        Rotations.XcwZ: Orientation.UP,
        Rotations.XccwZ: Orientation.UP,
        Rotations.YcwX: Orientation.RIGHT,
        Rotations.YccwX: Orientation.LEFT,
        Rotations.YcwZ: Orientation.BACKWARD,
        Rotations.YccwZ: Orientation.FORWARD,
        Rotations.ZcwX: Orientation.UP,
        Rotations.ZccwX: Orientation.UP,
        Rotations.ZcwY: Orientation.BACKWARD,
        Rotations.ZccwY: Orientation.FORWARD,
    },
    Orientation.DOWN: {
        Rotations.XcwY: Orientation.LEFT,
        Rotations.XccwY: Orientation.RIGHT,
        Rotations.XcwZ: Orientation.DOWN,
        Rotations.XccwZ: Orientation.DOWN,
        Rotations.YcwX: Orientation.LEFT,
        Rotations.YccwX: Orientation.RIGHT,
        Rotations.YcwZ: Orientation.FORWARD,
        Rotations.YccwZ: Orientation.BACKWARD,
        Rotations.ZcwX: Orientation.DOWN,
        Rotations.ZccwX: Orientation.DOWN,
        Rotations.ZcwY: Orientation.FORWARD,
        Rotations.ZccwY: Orientation.BACKWARD,
    },
    Orientation.LEFT: {
        Rotations.XcwY: Orientation.UP,
        Rotations.XccwY: Orientation.DOWN,
        Rotations.XcwZ: Orientation.BACKWARD,
        Rotations.XccwZ: Orientation.FORWARD,
        Rotations.YcwX: Orientation.UP,
        Rotations.YccwX: Orientation.DOWN,
        Rotations.YcwZ: Orientation.LEFT,
        Rotations.YccwZ: Orientation.LEFT,
        Rotations.ZcwX: Orientation.FORWARD,
        Rotations.ZccwX: Orientation.BACKWARD,
        Rotations.ZcwY: Orientation.LEFT,
        Rotations.ZccwY: Orientation.LEFT,
    },
    Orientation.RIGHT: {
        Rotations.XcwY: Orientation.DOWN,
        Rotations.XccwY: Orientation.UP,
        Rotations.XcwZ: Orientation.FORWARD,
        Rotations.XccwZ: Orientation.BACKWARD,
        Rotations.YcwX: Orientation.DOWN,
        Rotations.YccwX: Orientation.UP,
        Rotations.YcwZ: Orientation.RIGHT,
        Rotations.YccwZ: Orientation.RIGHT,
        Rotations.ZcwX: Orientation.BACKWARD,
        Rotations.ZccwX: Orientation.FORWARD,
        Rotations.ZcwY: Orientation.RIGHT,
        Rotations.ZccwY: Orientation.RIGHT,
    },
    Orientation.FORWARD: {
        Rotations.XcwY: Orientation.FORWARD,
        Rotations.XccwY: Orientation.FORWARD,
        Rotations.XcwZ: Orientation.RIGHT,
        Rotations.XccwZ: Orientation.LEFT,
        Rotations.YcwX: Orientation.FORWARD,
        Rotations.YccwX: Orientation.FORWARD,
        Rotations.YcwZ: Orientation.DOWN,
        Rotations.YccwZ: Orientation.UP,
        Rotations.ZcwX: Orientation.RIGHT,
        Rotations.ZccwX: Orientation.LEFT,
        Rotations.ZcwY: Orientation.UP,
        Rotations.ZccwY: Orientation.DOWN,
    },
    Orientation.BACKWARD: {
        Rotations.XcwY: Orientation.BACKWARD,
        Rotations.XccwY: Orientation.BACKWARD,
        Rotations.XcwZ: Orientation.LEFT,
        Rotations.XccwZ: Orientation.RIGHT,
        Rotations.YcwX: Orientation.BACKWARD,
        Rotations.YccwX: Orientation.BACKWARD,
        Rotations.YcwZ: Orientation.UP,
        Rotations.YccwZ: Orientation.DOWN,
        Rotations.ZcwX: Orientation.LEFT,
        Rotations.ZccwX: Orientation.RIGHT,
        Rotations.ZcwY: Orientation.DOWN,
        Rotations.ZccwY: Orientation.UP,
    },
}


def rotate_orientation(o: Orientation, rs: List[Rotations]) -> Orientation:
    for r in rs:
        o = _rotate_orientations[o][r]
    return o


class LSolver:
    def __init__(self,
                 atoms_alphabet: Dict[str, Dict[AtomAction, Any]],
                 parser: LParser):
        self.atoms_alphabet = atoms_alphabet
        self.parser = parser
        self.constraints = []

        self._calls = {
            AtomAction.PLACE: self._place,
            AtomAction.MOVE: self._move,
            AtomAction.ROTATE: self._rotate,
            AtomAction.PUSH: self._push,
            AtomAction.POP: self._pop
        }

        self.inner_loops_during = 5
        self.inner_loops_end = 5

    def _forward_expansion(self,
                           axiom: str,
                           n: int,
                           dc_check: bool = False) -> None:
        for i in range(n):
            axiom = self.parser.expand(axiom=axiom)
        if dc_check and len([c for c in self.constraints if c.when == ConstraintTime.DURING]) > 0:
            print('--- EXPANDING AXIOM ---\n', axiom)
            print('--- EXPANSION CONSTRAINTS CHECK ---')
            if not self._check_constraints(axiom=axiom,
                                           when=ConstraintTime.DURING)[ConstraintLevel.HARD_CONSTRAINT]:
                axiom = None  # do not continue expansion if it breaks hard constraints during expansion
        return axiom

    def _check_constraints(self,
                           axiom: str,
                           when: ConstraintTime) -> Dict[ConstraintLevel, bool]:
        sat = {
            ConstraintLevel.SOFT_CONSTRAINT: True,
            ConstraintLevel.HARD_CONSTRAINT: True,
        }
        for l in sat.keys():
            for c in self.constraints:
                if c.when == when and c.level == l:
                    s = c.constraint(axiom)
                    print(f'{c}:\t{s}')
                    sat[l] &= s
        return sat

    def solve(self,
              axiom: str,
              iterations: int,
              axioms_per_iteration: int = 1) -> str:
        all_axioms = [axiom[:]]
        # forward expansion + DURING constraints check
        dc_check = False
        for i in range(iterations):
            print(
                f'Expansion n.{i+1}/{iterations}; current number of axioms: {len(all_axioms)}')
            new_all_axioms = []
            for axiom in all_axioms:
                for _ in range(axioms_per_iteration):
                    new_axiom = axiom[:]
                    new_axiom = self._forward_expansion(axiom=new_axiom,
                                                        n=1,
                                                        dc_check=i > 0)
                    if new_axiom is not None:
                        new_all_axioms.append(new_axiom)
            all_axioms = new_all_axioms
            all_axioms = list(set(all_axioms))  # remove duplicates

        # END constraints check + possible backtracking
        if len([c for c in self.constraints if c.when == ConstraintTime.END]) > 0:
            to_rem = []
            for axiom in all_axioms:
                print(f'--- AXIOM ---\n', axiom)
                print('--- FINAL CONSTRAINTS CHECK---')
                if not self._check_constraints(axiom=axiom,
                                               when=ConstraintTime.END)[ConstraintLevel.HARD_CONSTRAINT]:
                    to_rem.append(axiom)
            # remaining axioms are SAT
            for a in to_rem:
                all_axioms.remove(a)

        # TODO: think of a better return choice
        if all_axioms:
            return all_axioms[0]
        else:
            raise Exception('No axiom could satisfy all HARD constraints')

    def add_constraint(self,
                       c: ConstraintHandler) -> None:
        self.constraints.append(c)

    class _ProcessData:
        def __init__(self,
                     position: Vec,
                     structure: Structure):
            self.position = position
            self.rotations = []
            self.position_history = []
            self.structure = structure
            # temporary, used only when extracting properties
            # (which will need a rework)
            self.a = ''
            self.i = 0
            self.axiom = ''

    def _rotate(self,
                data: _ProcessData,
                action_args: Any) -> int:
        data.rotations.append(action_args)
        return 0

    def _move(self,
              data: _ProcessData,
              action_args: Any) -> int:
        if data.rotations:
            action_args = rotate_orientation(action_args,
                                             data.rotations)
        data.position = data.position.sum(action_args.value)
        return 0

    def _push(self,
              data: _ProcessData,
              action_args: Any) -> int:
        data.position_history.append(data.position)
        return 0

    def _pop(self,
             data: _ProcessData,
             action_args: Any) -> int:
        data.position = data.position_history.pop(-1)
        if data.rotations:
            data.rotations.pop(-1)
        return 0

    def _place(self,
               data: _ProcessData,
               action_args: Any) -> int:
        orientation_forward, orientation_up = data.axiom[data.i + len(
            data.a):data.i + len(data.a) + 2]
        orientation_forward = orientations_ids[orientation_forward]
        orientation_up = orientations_ids[orientation_up]
        if data.rotations:
            orientation_forward = rotate_orientation(
                orientation_forward, data.rotations)
            orientation_up = rotate_orientation(orientation_up, data.rotations)
        data.structure.add_block(block=Block(block_type=action_args[0],
                                             orientation_forward=orientation_forward,
                                             orientation_up=orientation_up),
                                 grid_position=data.position.as_tuple())
        return 2

    def fill_structure(self,
                       structure: Structure,
                       axiom: str) -> None:
        data = self._ProcessData(position=Vec.v3i(0, 0, 0),
                                 structure=structure)
        data.axiom = axiom
        i = 0
        while i < len(axiom):
            for a in self.atoms_alphabet.keys():
                if axiom.startswith(a, i):
                    data.i = i
                    data.a = a
                    action, args = self.atoms_alphabet[a]['action'], self.atoms_alphabet[a]['args']
                    i += self._calls[action](data, args)
                    i += len(a)
                    break
