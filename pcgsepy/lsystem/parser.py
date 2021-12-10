from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List

from ..config import PL_LOW, PL_HIGH
from .rules import StochasticRules


class LParser(ABC):

    def __init__(self, rules: StochasticRules):
        self.rules = rules

    @abstractmethod
    def expand(self, axiom: str) -> str:
        pass


class HLParser(LParser):

    def expand(self, axiom: str) -> str:
        i = 0
        while i < len(axiom):
            offset = 0
            for k in self.rules.lhs_alphabet:
                if axiom[i:].startswith(k):
                    offset += len(k)
                    lhs = k
                    n = None
                    # check if there are parameters
                    if i + offset < len(axiom) and axiom[i + offset] == '(':
                        params = axiom[i +
                                       offset:axiom.index(')', i + offset + 1) +
                                       1]
                        offset += len(params)
                        n = int(params.replace('(', '').replace(')', ''))
                        lhs += '(x)'
                        if i + offset < len(axiom) and axiom[i + offset] == ']':
                            lhs += ']'
                            offset += 1
                    rhs = self.rules.get_rhs(lhs=lhs)
                    if n is not None or '(X)' in rhs:
                        # update rhs to include parameters
                        rhs = rhs.replace('(x)', f'({n})')
                        # TODO: get low-high values from config
                        rhs = rhs.replace(
                            '(X)', f'({np.random.randint(PL_LOW, PL_HIGH)})')
                    axiom = axiom[:i] + rhs + axiom[i + offset:]
                    i += len(rhs) - 1
                    break
            i += 1
        return axiom


class HLtoMLTranslator:

    def __init__(self, alphabet: Dict[str, Any], tiles_dims: Dict[str, Any],
                 tiles_block_offset: Dict[str, Any]):
        self.alphabet = alphabet
        self.td = tiles_dims
        self.tbo = tiles_block_offset

    def _axiom_as_list(self, axiom: str) -> List[Dict[str, Any]]:
        atoms_list = []
        i = 0
        while i < len(axiom):
            offset = 0
            for k in self.alphabet.keys():
                if axiom[i:].startswith(k):
                    offset += len(k)
                    n = None
                    # check if there are parameters (multiplicity)
                    if i + offset < len(axiom) and axiom[i + offset] == '(':
                        params = axiom[i +
                                       offset:axiom.index(')', i + offset + 1) +
                                       1]
                        offset += len(params)
                        n = int(params.replace('(', '').replace(')', ''))
                    atoms_list.append({'atom': k})
                    if k in self.td.keys():
                        atoms_list[-1]['n'] = n if n is not None else 1

                    i += len(k) + (len(f'({n})') if n is not None else 0) - 1
                    break
            i += 1
        return atoms_list

    def _to_midlvl(self, atoms_list: List[Dict[str, Any]]) -> str:
        last_parents = []
        new_axiom = ''

        for i, atom in enumerate(atoms_list):
            a = atom['atom']
            n = atom.get('n', None)
            # tile dimensions (either current or parent's)
            if n is None:
                dims = self.td[last_parents[-1]]
            else:
                dims = self.td[a]
            # Translate w/ correction to mid-level
            # Placeable tile
            if a in self.td.keys():
                new = [f"{a}>({dims.z})" for _ in range(n)]
                new_axiom += ''.join(new)
            # Position stack manipulation
            elif a == '[' or a == ']':
                new_axiom += a
            # Rotation
            elif a.startswith('Rot'):
                next_tile = None
                for j in range(i + 1, len(atoms_list)):
                    if atoms_list[j].get('n', None) is not None:
                        next_tile = atoms_list[j]['atom']
                        break
                next_dims = self.td[next_tile]
                next_offset = self.tbo[next_tile]
                c = ''
                if a == 'RotZccwX':
                    c = f"+({dims.x})>({next_dims.x - dims.z})"
                elif a == 'RotZcwX':
                    c = f"-({next_offset})"
                elif a == 'RotZcwY':
                    c = f"?({self.tbo[a]})"
                elif a == 'RotZccwY':
                    c = f"!({dims.y})>({next_dims.y - dims.z})"
                elif a == 'RotXcwY':
                    c = f"?({next_offset})"
                elif a == 'RotXccwY':
                    c = f"-({next_offset})"
                elif a == 'RotXcwZ':
                    c = f"-({next_offset})"
                elif a == 'RotXccwZ':
                    c = f"+({dims.x})>({next_dims.x - dims.z})"
                elif a == 'RotYcwX':
                    c = f"?({next_offset})"
                elif a == 'RotYccwX':
                    c = f"-({next_offset})"
                elif a == 'RotYcwZ':
                    c = f"?({next_offset})"
                elif a == 'RotYccwZ':
                    c = f"!({dims.y})>({next_dims.y - dims.z})"
                new_axiom += ''.join([c, a])

            # Parent's stack manipulation
            if i + 1 < len(atoms_list):
                if a != ']' and atoms_list[i + 1]['atom'] == '[':
                    last_parents.append(a)
                if a == ']' and atoms_list[i + 1].get('n', None) is not None:
                    last_parents.pop(-1)

        return new_axiom

    def transform(self, axiom: str) -> str:
        atoms_list = self._axiom_as_list(axiom)
        return self._to_midlvl(atoms_list)


class LLParser(LParser):

    def expand(self, axiom: str) -> str:
        i = 0
        while i < len(axiom):
            for k in self.rules.lhs_alphabet:
                if axiom[i:].startswith(k):
                    rhs = self.rules.get_rhs(lhs=k)
                    axiom = axiom[:i] + rhs + axiom[i + len(k):]
                    i += len(rhs) - 1
                    break
            i += 1
        return axiom