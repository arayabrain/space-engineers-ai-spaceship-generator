from typing import Dict

from .actions import *
from .constraints import *
from .parser import LParser
from ..structure import *
from ..common.vecs import orientation_from_str


class LSolver:
    def __init__(self,
                 atoms_alphabet: Dict[str, Dict[AtomAction, Any]],
                 parser: LParser):
        self.atoms_alphabet = atoms_alphabet
        self.parser = parser
        self.constraints = []
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
                    s = c.constraint(axiom=axiom,
                                     extra_args=c.extra_args)
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
        c.extra_args['alphabet'] = self.atoms_alphabet
