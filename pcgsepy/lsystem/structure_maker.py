from ..structure import Structure, Block
from ..common.vecs import Vec, orientation_from_str, orientation_from_vec
from .actions import rotation_matrices, AtomAction

from abc import ABC, abstractmethod
from typing import Any, Dict


class StructureMaker(ABC):

    def __init__(self, atoms_alphabet, position: Vec):
        self.atoms_alphabet = atoms_alphabet
        self._calls = {
            AtomAction.PLACE: self._place,
            AtomAction.MOVE: self._move,
            AtomAction.ROTATE: self._rotate,
            AtomAction.PUSH: self._push,
            AtomAction.POP: self._pop
        }
        self.position = position
        self.rotations = []
        self.position_history = []

    def _apply_rotation(self, arr: Vec) -> Vec:
        arr = arr.as_array()
        for rot in reversed(self.rotations):
            arr = rot.dot(arr)
        return Vec.from_np(arr)

    def _rotate(self, action_args: Any) -> None:
        self.rotations.append(rotation_matrices[action_args['action_args']])

    def _move(self, action_args: Any) -> None:
        dpos = action_args['action_args'].value
        n = int(action_args['parameters'][0])
        if self.rotations:
            dpos = self._apply_rotation(arr=dpos)
        for _ in range(n):
            self.position = self.position.sum(dpos)

    def _push(self, action_args: Any) -> None:
        self.position_history.append(self.position)

    def _pop(self, action_args: Any) -> None:
        self.position = self.position_history.pop(-1)
        if self.rotations:
            self.rotations.pop(-1)

    @abstractmethod
    def _place(self, action_args: Any) -> None:
        pass

    @abstractmethod
    def fill_structure(self,
                       structure: Structure,
                       axiom: str,
                       additional_args: Dict[str, Any] = {}) -> None:
        pass


class LLStructureMaker(StructureMaker):

    def _place(self, action_args: Any) -> None:
        orientation_forward, orientation_up = action_args['parameters'][
            0], action_args['parameters'][1]
        orientation_forward = orientation_from_str[orientation_forward]
        orientation_up = orientation_from_str[orientation_up]
        if self.rotations:
            orientation_forward = orientation_from_vec(
                self._apply_rotation(arr=orientation_forward.value))
            orientation_up = orientation_from_vec(
                self._apply_rotation(arr=orientation_up.value))
        block = Block(block_type=action_args['action_args'][0],
                      orientation_forward=orientation_forward,
                      orientation_up=orientation_up)
        self.structure.add_block(
            block=block,
            grid_position=self.position.as_tuple(),
            exit_on_duplicates=action_args['intersection_checking'])

    def fill_structure(self,
                       structure: Structure,
                       axiom: str,
                       additional_args: Dict[str, Any] = {}) -> Structure:
        self.additional_args = additional_args
        self.structure = structure
        intersection_checking = additional_args.get('intersection_checking',
                                                    False)
        i = 0
        while i < len(axiom):
            offset = 0
            for a in reversed(self.atoms_alphabet.keys()):
                if axiom.startswith(a, i):
                    offset += len(a)
                    # check for atom's parameters
                    parameters = []
                    if i + offset < len(axiom) and axiom[i + offset] == '(':
                        params = axiom[i +
                                       offset:axiom.index(')', i + offset + 1) +
                                       1]
                        offset += len(params)
                        parameters = params.replace('(',
                                                    '').replace(')',
                                                                '').split(',')
                    action, args = self.atoms_alphabet[a][
                        'action'], self.atoms_alphabet[a]['args']
                    self._calls[action]({
                        'action_args': args,
                        'parameters': parameters,
                        'axiom': a,
                        'intersection_checking': intersection_checking
                    })
                    break
            i += offset
        return self.structure
