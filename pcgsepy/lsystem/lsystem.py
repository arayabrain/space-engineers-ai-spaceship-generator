import logging
from typing import List, Optional, Tuple

from ..common.vecs import Orientation, Vec
from ..config import N_APE
from .constraints import ConstraintHandler
from .solver import LSolver
from .structure_maker import LLStructureMaker
from ..structure import Structure


class LSystemModule:

    def __init__(self, hl_solver: LSolver, ll_solver: LSolver, name: str):
        self.hlsolver = hl_solver
        self.llsolver = ll_solver
        self.check_sat = True
        self.name = name
        self.hl_constraints = []
        self.ll_constraints = []

    def add_hl_constraint(self, c: ConstraintHandler) -> None:
        self.hl_constraints.append(c)

    def add_ll_constraint(self, c: ConstraintHandler) -> None:
        self.ll_constraints.append(c)

    def get_hl_axioms(self,
                      starting_axiom: str,
                      iterations: int = 1) -> List[str]:
        return self.hlsolver.solve(axiom=starting_axiom,
                                   iterations=iterations,
                                   axioms_per_iteration=N_APE,
                                   check_sat=self.check_sat)

    def get_ml_axioms(self, hl_axioms: List[str]) -> List[str]:
        return [
            self.hlsolver.translator.transform(axiom=hl_axiom)
            for hl_axiom in hl_axioms
        ]

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

    def apply_rules(
        self,
        starting_axiom: str,
        iterations: int = 1,
        make_graph: bool = False
    ) -> Tuple[List[Structure], List[str], List[str]]:
        self.hlsolver.set_constraints(cs=self.hl_constraints)
        self.llsolver.set_constraints(cs=self.ll_constraints)

        logging.getLogger('base-logger').info(
            f'[{self.name}] Started high level solving...')
        hl_axioms = self.get_hl_axioms(starting_axiom=starting_axiom,
                                       iterations=iterations)

        logging.getLogger('base-logger').debug(
            f'[{self.name}] Converting HL axioms to ML...')
        ml_axioms = self.get_ml_axioms(hl_axioms)

        logging.getLogger('base-logger').info(
            f'[{self.name}] Started low level solving...')
        ll_axioms, to_rem = self.get_ll_axioms(ml_axioms)

        for i in reversed(to_rem):
            hl_axioms.pop(i)

        return hl_axioms


class LSystem:

    def __init__(self, hl_solver: LSolver, ll_solver: LSolver,
                 names: List[str]):
        self.hl_solver = hl_solver
        self.ll_solver = ll_solver
        self.check_sat = True
        self.modules = [
            LSystemModule(hl_solver=hl_solver, ll_solver=ll_solver, name=name)
            for name in names
        ]
        self.all_hl_constraints = set()
        self.all_ll_constraints = set()

    def enable_sat_check(self):
        self.check_sat = True
        for m in self.modules:
            m.check_sat = True

    def disable_sat_check(self):
        self.check_sat = False
        for m in self.modules:
            m.check_sat = False

    # add own constraints as well
    def add_hl_constraints(self,
                           cs: List[List[Optional[ConstraintHandler]]]) -> None:
        assert len(cs) == len(
            self.modules
        ), f'Wrong number of expected modules: have {len(self.modules)}, passed {len(cs)}.'
        for m, mcs in zip(self.modules, cs):
            for c in mcs:
                m.add_hl_constraint(c)
                self.all_hl_constraints.add(c)

    # add own constraints as well
    def add_ll_constraints(self,
                           cs: List[List[Optional[ConstraintHandler]]]) -> None:
        assert len(cs) == len(
            self.modules
        ), f'Wrong number of expected modules: have {len(self.modules)}, passed {len(cs)}.'
        for m, mcs in zip(self.modules, cs):
            for c in mcs:
                m.add_ll_constraint(c)
                self.all_ll_constraints.add(c)

    def process_module(self,
                       module: LSystemModule,
                       starting_axiom: str,
                       iterations: int = 1) -> List[str]:
        return module.apply_rules(starting_axiom=starting_axiom,
                                  iterations=iterations)

    def _merge_axioms(self, axioms: List[str]) -> str:
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

    def _produce_axioms_combinations(self,
                                     m_axioms: List[List[str]]) -> List[str]:
        import itertools
        # Cartesian product of all axioms, return merged axiom
        return [
            self._merge_axioms(x) for x in list(itertools.product(*m_axioms))
        ]

    def apply_rules(
        self,
        starting_axioms: List[str],
        iterations: List[int],
        create_structures: bool = False,
        make_graph: bool = False
    ) -> Tuple[List[Structure], List[str], List[str]]:
        assert len(starting_axioms) == len(
            self.modules
        ), f'Assumed wrong number of modules: have {len(self.modules)}, passed {len(starting_axioms)}.'
        assert len(iterations) == len(
            self.modules
        ), f'Assumed wrong number of modules: have {len(self.modules)}, passed {len(iterations)}.'

        all_axioms = []
        for module, starting_axiom, n_iterations in zip(self.modules,
                                                        starting_axioms,
                                                        iterations):
            hl_axioms = self.process_module(module=module,
                                            starting_axiom=starting_axiom,
                                            iterations=n_iterations)
            all_axioms.append(hl_axioms)

        all_axioms = self._produce_axioms_combinations(m_axioms=all_axioms)
        ml_axioms = [
            self.hl_solver.translator.transform(axiom=hl_axiom)
            for hl_axiom in all_axioms
        ]
        ll_axioms = [
            self.ll_solver.solve(axiom=ml_axiom,
                                 iterations=1,
                                 axioms_per_iteration=1,
                                 check_sat=False)[0] for ml_axiom in ml_axioms
        ]

        structures = []
        if create_structures:
            base_position, orientation_forward, orientation_up = Vec.v3i(
                0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
            for i, ll_axiom in enumerate(ll_axioms):
                structure = Structure(origin=base_position,
                                      orientation_forward=orientation_forward,
                                      orientation_up=orientation_up)
                structure = LLStructureMaker(
                    atoms_alphabet=self.ll_solver.atoms_alphabet,
                    position=base_position).fill_structure(structure=structure,
                                                           axiom=ll_axiom)
                structure.sanify()
                structures.append(structure)
                if make_graph:
                    structure.show(title=all_axioms[i])

        return structures, all_axioms, ll_axioms
