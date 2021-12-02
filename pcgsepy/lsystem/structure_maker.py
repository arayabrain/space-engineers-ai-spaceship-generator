from ..structure import Structure, Block
from ..common.vecs import Vec, orientation_from_str, Orientation, orientation_from_vec
from .actions import rotation_matrices, AtomAction
from .constraints import build_polyhedron, HLStructure

from typing import Any, Dict

class StructureMaker:
    def __init__(self,
                 atoms_alphabet,
                 position: Vec):
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
        # temporary, used only when extracting properties
        # (which will need a rework)
        self.a = ''
        self.i = 0
        self.axiom = ''
    
    def _apply_rotation(self,
                        arr: Vec) -> Vec:
        arr = arr.as_array()
        for rot in reversed(self.rotations):
                arr = rot.dot(arr)
        return Vec.from_np(arr)
        
    def _rotate(self,
                action_args: Any) -> int:
        self.rotations.append(rotation_matrices[action_args])
        return 0

    def _move(self,
              action_args: Any) -> int:
        action_args = action_args.value
        if self.rotations:
            action_args = self._apply_rotation(arr=action_args)
        self.position = self.position.sum(action_args)
        return 0

    def _push(self,
              action_args: Any) -> int:
        self.position_history.append(self.position)
        return 0

    def _pop(self,
             action_args: Any) -> int:
        self.position = self.position_history.pop(-1)
        if self.rotations:
            self.rotations.pop(-1)
        return 0

    def _place(self,
               action_args: Any) -> int:
        if type(self.structure) == Structure:
            orientation_forward, orientation_up = self.axiom[self.i + len(self.a):self.i + len(self.a) + 2]
            orientation_forward = orientation_from_str[orientation_forward]
            orientation_up = orientation_from_str[orientation_up]
            if self.rotations:
                orientation_forward = orientation_from_vec(self._apply_rotation(arr=orientation_forward.value))
                orientation_up = orientation_from_vec(self._apply_rotation(arr=orientation_up.value))
            self.structure.add_block(block=Block(block_type=action_args[0],
                                                 orientation_forward=orientation_forward,
                                                 orientation_up=orientation_up),
                                     grid_position=self.position.as_tuple())
            return 2
        else:
            dims = self.additional_args['tiles_dimensions'][self.a].as_array()
            for r in reversed(self.rotations):
                dims = r.dot(dims)
            p = build_polyhedron(position=self.position,
                                 dims=Vec.from_np(dims))
            self.structure.add_hl_poly(p)
            return 0

    def fill_structure(self,
                       structure: Structure,
                       axiom: str,
                       additional_args: Dict[str, Any] = {}) -> None:
        self.axiom = axiom
        self.additional_args = additional_args
        self.structure = structure if structure else HLStructure()
        i = 0
        while i < len(axiom):
            for a in self.atoms_alphabet.keys():
                if axiom.startswith(a, i):
                    self.i = i
                    self.a = a
                    action, args = self.atoms_alphabet[a]['action'], self.atoms_alphabet[a]['args']
                    i += self._calls[action](args)
                    i += len(a)
                    break
        return self.structure