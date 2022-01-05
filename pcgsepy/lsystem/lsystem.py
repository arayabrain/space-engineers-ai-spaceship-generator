import logging
from typing import List, Tuple

from ..common.vecs import Orientation, Vec
from ..config import N_APE
from .constraints import ConstraintHandler
from .solver import LSolver
from .structure_maker import LLStructureMaker
from ..structure import Structure


class LSystem:

    def __init__(self, hl_solver: LSolver, ll_solver: LSolver):
        self.hlsolver = hl_solver
        self.llsolver = ll_solver
        self.check_sat = True

    def add_hl_constraint(self, c: ConstraintHandler) -> None:
        self.hlsolver.add_constraint(c)

    def add_ll_constraint(self, c: ConstraintHandler) -> None:
        self.llsolver.add_constraint(c)

    def get_hl_axioms(self,
                      starting_axiom: str,
                      iterations: int = 1) -> List[str]:
        return self.hlsolver.solve(axiom=starting_axiom,
                                   iterations=iterations,
                                   axioms_per_iteration=N_APE,
                                   check_sat=self.check_sat)

    def get_ml_axioms(self,
                      hl_axioms: List[str]) -> List[str]:
        return [self.hlsolver.translator.transform(axiom=hl_axiom) for hl_axiom in hl_axioms]

    def get_ll_axioms(self,
                      ml_axioms: List[str]) -> Tuple[List[str], List[str]]:
        ll_axioms, to_rem = [], []
        for i, ml_axiom in enumerate(ml_axioms):
            ll_axiom = self.llsolver.solve(axiom=ml_axiom,
                                           iterations=1,
                                           axioms_per_iteration=1,
                                           check_sat=self.check_sat)
            if ll_axiom:
                ll_axioms.extend(ll_axiom)
            else:
                to_rem.append(i)
        return ll_axioms, to_rem

    def apply_rules(self,
                    starting_axiom: str,
                    iterations: int = 1,
                    make_graph: bool = False) -> Tuple[List[Structure], List[str], List[str]]:
        base_position, orientation_forward, orientation_up = Vec.v3i(
            0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
        structure = Structure(origin=base_position,
                              orientation_forward=orientation_forward,
                              orientation_up=orientation_up)
        logging.getLogger('base-logger').info('Started high level solving...')
        hl_axioms = self.get_hl_axioms(starting_axiom=starting_axiom,
                                       iterations=iterations)

        logging.getLogger('base-logger').debug('Converting HL axioms to ML...')
        ml_axioms = self.get_ml_axioms(hl_axioms)

        logging.getLogger('base-logger').info('Started low level solving...')
        ll_axioms, to_rem = self.get_ll_axioms(ml_axioms)

        for i in reversed(to_rem):
            hl_axioms.pop(i)

        structures = []
        for i, ll_axiom in enumerate(ll_axioms):
            structure = LLStructureMaker(
                atoms_alphabet=self.llsolver.atoms_alphabet,
                position=base_position).fill_structure(structure=structure,
                                                       axiom=ll_axiom)
            structure.sanify()
            structures.append(structure)
            if make_graph:
                structure.show(title=hl_axioms[i])

        return structures, hl_axioms, ll_axioms


def merge_axioms(axioms: List[str]) -> str:
    """
    Merge axioms in a single axiom.

    Parameters
    ----------
    axioms : List[str]
        The list of axioms to merge, ordered.

    Returns
    -------
    str
        The merged axiom
    """
    # any additional control on alignment etc. should
    # be done here.
    return ''.join(axioms)