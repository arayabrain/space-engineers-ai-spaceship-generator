import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..common.vecs import Orientation, Vec
from ..config import N_SPE
from .constraints import ConstraintHandler
from .solver import LSolver
from .structure_maker import LLStructureMaker
from ..structure import Structure
from .solution import CandidateSolution, merge_solutions


class LSystemModule:

    def __init__(self,
                 hl_solver: LSolver,
                 ll_solver: LSolver,
                 name: str):
        self.hlsolver = hl_solver
        self.llsolver = ll_solver
        self.check_sat = True
        self.name = name
        self.active = True
        self.hl_constraints = []
        self.ll_constraints = []

    def add_hl_constraint(self,
                          c: ConstraintHandler) -> None:
        self.hl_constraints.append(c)

    def add_ll_constraint(self,
                          c: ConstraintHandler) -> None:
        self.ll_constraints.append(c)

    def get_hl_solutions(self,
                         starting_string: str,
                         iterations: int = 1) -> List[CandidateSolution]:
        return self.hlsolver.solve(string=starting_string,
                                   iterations=iterations,
                                   strings_per_iteration=N_SPE,
                                   check_sat=self.check_sat)

    def get_ml_strings(self,
                       hl_strings: List[str]) -> List[str]:
        return [
            self.hlsolver.translator.transform(string=hl_string)
            for hl_string in hl_strings
        ]

    def get_ll_solutions(self,
                         cs: List[CandidateSolution]) -> Tuple[List[CandidateSolution], npt.NDArray[np.bool8]]:
        ll_solutions, to_keep = [], np.zeros(shape=len(cs), dtype=np.bool8)
        for i, cs in enumerate(cs):
            ll_solution = self.llsolver.solve(string=cs.string,
                                              iterations=1,
                                              strings_per_iteration=1,
                                              check_sat=self.check_sat)[0]
            if ll_solution:
                ll_solutions.append(ll_solution)
                to_keep[i] = True
        return ll_solutions, to_keep

    def apply_rules(self,
                    starting_string: str,
                    iterations: int = 1) -> List[CandidateSolution]:
        self.hlsolver.set_constraints(cs=self.hl_constraints)
        self.llsolver.set_constraints(cs=self.ll_constraints)

        logging.getLogger('base-logger').info(f'[{self.name}] Started high level solving...')
        hl_solutions = self.get_hl_solutions(starting_string=starting_string,
                                             iterations=iterations)

        logging.getLogger('base-logger').debug(f'[{self.name}] Converting HL strings to ML...')
        ml_strings = self.get_ml_strings([cs.string for cs in hl_solutions])

        logging.getLogger('base-logger').info(f'[{self.name}] Started low level solving...')
                
        _, to_keep = self.get_ll_solutions([CandidateSolution(string=s,
                                                              content=hl_cs._content) for s, hl_cs in zip(ml_strings, hl_solutions)])

        hl_solutions = [x for x, k in zip(hl_solutions, to_keep) if k]
        
        return hl_solutions
    
    def to_json(self) -> Dict[str, Any]:
        return {
            'hlsolver': self.hlsolver.to_json(),
            'llsolver': self.llsolver.to_json(),
            'check_sat': self.check_sat,
            'name': self.name,
            'active': self.active,
            'hl_constraints': [c.to_json() for c in self.hl_constraints],
            'll_constraints': [c.to_json() for c in self.ll_constraints],
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'LSystemModule':
        lsm = LSystemModule(hl_solver=LSolver.from_json(my_args['hlsolver']),
                            ll_solver=LSolver.from_json(my_args['llsolver']),
                            name=my_args['name'])
        lsm.active = my_args['active']
        lsm.hl_constraints = [ConstraintHandler.from_json(c) for c in my_args['hl_constraints']]
        lsm.ll_constraints = [ConstraintHandler.from_json(c) for c in my_args['ll_constraints']]
        return lsm
        


class LSystem:

    def __init__(self,
                 hl_solver: LSolver,
                 ll_solver: LSolver,
                 names: List[str]):
        self.hl_solver = hl_solver
        self.ll_solver = ll_solver
        self.check_sat = True
        self.modules = [
            LSystemModule(hl_solver=hl_solver,
                          ll_solver=ll_solver,
                          name=name)
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

    def add_hl_constraints(self,
                           cs: List[List[Optional[ConstraintHandler]]]) -> None:
        assert len(cs) == len(self.modules), f'Wrong number of expected modules: have {len(self.modules)}, passed {len(cs)}.'
        for m, mcs in zip(self.modules, cs):
            for c in mcs:
                m.add_hl_constraint(c)
                self.all_hl_constraints.add(c)

    def add_ll_constraints(self,
                           cs: List[List[Optional[ConstraintHandler]]]) -> None:
        assert len(cs) == len(self.modules), f'Wrong number of expected modules: have {len(self.modules)}, passed {len(cs)}.'
        for m, mcs in zip(self.modules, cs):
            for c in mcs:
                m.add_ll_constraint(c)
                self.all_ll_constraints.add(c)

    def hl_to_ll(self,
                 cs: CandidateSolution) -> CandidateSolution:
        return self.ll_solver.solve(string=self.hl_solver.translator.transform(string=cs.string),
                                    iterations=1,
                                    check_sat=False)[0]

    def process_module(self,
                       module: LSystemModule,
                       starting_string: str,
                       iterations: int = 1) -> List[CandidateSolution]:
        return module.apply_rules(starting_string=starting_string,
                                  iterations=iterations)

    def _produce_strings_combinations(self,
                                      lcs: List[CandidateSolution]) -> List[CandidateSolution]:
        """Produce the combination of solutions' strings.

        Args:
            lcs (List[CandidateSolution]): The list of solutions

        Returns:
            List[CandidateSolution]: The list of solutions
        """
        import itertools
        # Cartesian product of all strings, return merged string
        modules_names = [m.name for m in self.modules]
        return [merge_solutions(lcs=x, modules_names=modules_names) for x in list(itertools.product(*lcs))]

    def apply_rules(self,
                    starting_strings: List[str],
                    iterations: List[int],
                    create_structures: bool = False,
                    make_graph: bool = False) -> List[CandidateSolution]:
        assert len(starting_strings) == len(self.modules), f'Assumed wrong number of modules: have {len(self.modules)}, passed {len(starting_strings)}.'
        assert len(iterations) == len(self.modules), f'Assumed wrong number of modules: have {len(self.modules)}, passed {len(iterations)}.'
        # create solutions for each module
        solutions = []
        for module, starting_string, n_iterations in zip(self.modules,
                                                         starting_strings,
                                                         iterations):
            lcs = self.process_module(module=module,
                                      starting_string=starting_string,
                                      iterations=n_iterations)
            solutions.append(lcs)
        # combine module's solutions
        solutions = self._produce_strings_combinations(lcs=solutions)
        # set mid- and low-level strings
        for cs in solutions:
            ml_string = self.hl_solver.translator.transform(string=cs.string)
            cs.ll_string = self.ll_solver.solve(string=ml_string,
                                                iterations=1,
                                                strings_per_iteration=1,
                                                check_sat=False)[0].string
            # add content to solution if enabled
            if create_structures:
                base_position, orientation_forward, orientation_up = Vec.v3i(
                    0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
                structure = Structure(origin=base_position,
                                      orientation_forward=orientation_forward,
                                      orientation_up=orientation_up)
                structure = LLStructureMaker(
                    atoms_alphabet=self.ll_solver.atoms_alphabet,
                    position=base_position).fill_structure(structure=structure,
                                                           string=cs.ll_string)
                structure.sanify()
                cs.set_content(content=structure)
                if make_graph:
                    structure.show(title=cs.string)

        return solutions

    def to_json(self) -> Dict[str, Any]:
        return {
            'hlsolver': self.hl_solver.to_json(),
            'llsolver': self.ll_solver.to_json(),
            'check_sat': self.check_sat,
            'modules': [m.to_json() for m in self.modules],
            'all_hl_constraints': [c.to_json() for c in self.all_hl_constraints],
            'all_ll_constraints': [c.to_json() for c in self.all_ll_constraints]
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'LSystem':
        ls = LSystem(hl_solver=LSolver.from_json(my_args['hlsolver']),
                     ll_solver=LSolver.from_json(my_args['llsolver']),
                     names=[])
        ls.modules = [LSystemModule.from_json(lsm) for lsm in my_args['modules']]
        ls.all_hl_constraints = set([ConstraintHandler.from_json(c) for c in my_args['all_hl_constraints']])
        ls.all_ll_constraints = set([ConstraintHandler.from_json(c) for c in my_args['all_ll_constraints']])
        return ls