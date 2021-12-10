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

    def add_hl_constraint(self, c: ConstraintHandler) -> None:
        self.hlsolver.add_constraint(c)

    def add_ll_constraint(self, c: ConstraintHandler) -> None:
        self.llsolver.add_constraint(c)

    def apply_rules(self,
                    starting_axiom: str,
                    iterations: int = 1,
                    make_graph: bool = False) -> Structure:
        base_position, orientation_forward, orientation_up = Vec.v3i(
            0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
        structure = Structure(origin=base_position,
                              orientation_forward=orientation_forward,
                              orientation_up=orientation_up)
        print('### HL SOLVER ###')
        hl_axiom = self.hlsolver.solve(axiom=starting_axiom,
                                       iterations=iterations,
                                       axioms_per_iteration=N_APE)

        translator = self.hlsolver.translator
        ml_axiom = translator.transform(axiom=hl_axiom)
        print(f'\n-- FINAL AXIOM TRANSLATION --\n{hl_axiom} -> {ml_axiom}')

        print('\n### LL SOLVER ###')
        ll_axiom = self.llsolver.solve(axiom=ml_axiom, iterations=1)

        structure = LLStructureMaker(
            atoms_alphabet=self.llsolver.atoms_alphabet,
            position=base_position).fill_structure(structure=structure,
                                                   axiom=ll_axiom)

        structure.sanify()

        if make_graph:
            structure.show(title=hl_axiom)

        return structure, hl_axiom, ll_axiom
