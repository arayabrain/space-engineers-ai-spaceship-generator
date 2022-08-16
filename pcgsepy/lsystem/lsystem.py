import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pcgsepy.common.vecs import Orientation, Vec
from pcgsepy.config import N_SPE
from pcgsepy.lsystem.constraints import ConstraintHandler
from pcgsepy.lsystem.solution import CandidateSolution, merge_solutions
from pcgsepy.lsystem.solver import LSolver
from pcgsepy.lsystem.structure_maker import LLStructureMaker
from pcgsepy.structure import Structure


class LSystemModule:
    __slots__ = ['hlsolver', 'llsolver', 'check_sat',
                 'name', 'active', 'hl_constraints', 'll_constraints']

    def __init__(self,
                 hl_solver: LSolver,
                 ll_solver: LSolver,
                 name: str):
        """Create a module of the hierarchical L-system.

        Args:
            hl_solver (LSolver): The high-level solver.
            ll_solver (LSolver): The low-level solver.
            name (str): The name of the module.
        """
        self.hlsolver = hl_solver
        self.llsolver = ll_solver
        self.check_sat = True
        self.name = name
        self.active = True
        self.hl_constraints = []
        self.ll_constraints = []

    def add_hl_constraint(self,
                          c: ConstraintHandler) -> None:
        """Add a high-level constraint to the module.

        Args:
            c (ConstraintHandler): The constraint.
        """
        self.hl_constraints.append(c)

    def add_ll_constraint(self,
                          c: ConstraintHandler) -> None:
        """Add a low-level constraint to the module.

        Args:
            c (ConstraintHandler): The constraint.
        """
        self.ll_constraints.append(c)

    def _get_hl_solutions(self,
                          starting_string: str,
                          iterations: int = 1) -> List[CandidateSolution]:
        """Get the high-level solutions.

        Args:
            starting_string (str): The starting string.
            iterations (int, optional): The number of iterations to expand for. Defaults to 1.

        Returns:
            List[CandidateSolution]: The list of solutions.
        """
        return self.hlsolver.solve(string=starting_string,
                                   iterations=iterations,
                                   strings_per_iteration=N_SPE,
                                   check_sat=self.check_sat)

    def _get_ml_strings(self,
                        hl_strings: List[str]) -> List[str]:
        """Convert the high-level strings to mid-level.

        Args:
            hl_strings (List[str]): The high-level strings.

        Returns:
            List[str]: The list of mid-level strings.
        """
        return [self.hlsolver.translator.transform(string=hl_string) for hl_string in hl_strings]

    def _get_ll_solutions(self,
                          cs: List[CandidateSolution]) -> Tuple[List[CandidateSolution], npt.NDArray[np.bool8]]:
        """Get the low-level solutions.
        Make sure that each high-level solutions already has a mid-level string assigned.

        Args:
            cs (List[CandidateSolution]): The list of high-level solutions.

        Returns:
            Tuple[List[CandidateSolution], npt.NDArray[np.bool8]]: The list of solutions and which solutions should be kept.
        """
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
        """Apply the rules of the module to a given starting string.

        Args:
            starting_string (str): The starting string.
            iterations (int, optional): The number of iterations to expand for. Defaults to 1.

        Returns:
            List[CandidateSolution]: The list of solutions.
        """
        self.hlsolver.set_constraints(cs=self.hl_constraints)
        self.llsolver.set_constraints(cs=self.ll_constraints)
        logging.getLogger('base-logger').info(f'[{self.name}] Started high level solving...')
        hl_solutions = self._get_hl_solutions(starting_string=starting_string,
                                              iterations=iterations)
        logging.getLogger('base-logger').debug(f'[{self.name}] Converting HL strings to ML...')
        ml_strings = self._get_ml_strings([cs.string for cs in hl_solutions])
        logging.getLogger('base-logger').info(f'[{self.name}] Started low level solving...')
        _, to_keep = self._get_ll_solutions([CandidateSolution(string=s,
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
        """Create the hierarchical L-system. Each name corresponds to a L-system sub-module.

        Args:
            hl_solver (LSolver): The high-level solver.
            ll_solver (LSolver): The low-level solver.
            names (List[str]): The name of the modules.
        """
        self.hl_solver = hl_solver
        self.ll_solver = ll_solver
        self.check_sat = True
        self.modules = [LSystemModule(hl_solver=hl_solver,
                                      ll_solver=ll_solver,
                                      name=name) for name in names]
        self.all_hl_constraints = set()
        self.all_ll_constraints = set()

    def enable_sat_check(self):
        """Enable constraints satisfaction"""
        self.check_sat = True
        for m in self.modules:
            m.check_sat = True

    def disable_sat_check(self):
        """Disable constraints satisfaction"""
        self.check_sat = False
        for m in self.modules:
            m.check_sat = False

    def add_hl_constraints(self,
                           cs: List[List[Optional[ConstraintHandler]]]) -> None:
        """Add high-level constraints to each module.

        Args:
            cs (List[List[Optional[ConstraintHandler]]]): The list of constraints.
        """
        assert len(cs) == len(self.modules), f'Wrong number of expected modules: have {len(self.modules)}, passed {len(cs)}.'
        for m, mcs in zip(self.modules, cs):
            for c in mcs:
                m.add_hl_constraint(c)
                self.all_hl_constraints.add(c)

    def add_ll_constraints(self,
                           cs: List[List[Optional[ConstraintHandler]]]) -> None:
        """Add low-level constraints to each module.

        Args:
            cs (List[List[Optional[ConstraintHandler]]]): The list of constraints.
        """
        assert len(cs) == len(self.modules), f'Wrong number of expected modules: have {len(self.modules)}, passed {len(cs)}.'
        for m, mcs in zip(self.modules, cs):
            for c in mcs:
                m.add_ll_constraint(c)
                self.all_ll_constraints.add(c)

    def process_module(self,
                       module: LSystemModule,
                       starting_string: str,
                       iterations: int = 1) -> List[CandidateSolution]:
        """Process the given module, applying the expansion rules.

        Args:
            module (LSystemModule): The module.
            starting_string (str): The starting string.
            iterations (int, optional): The number of iterations to expand for. Defaults to 1.

        Returns:
            List[CandidateSolution]: The list of solutions.
        """
        return module.apply_rules(starting_string=starting_string,
                                  iterations=iterations)

    def _produce_solutions_combinations(self,
                                        lcs: List[CandidateSolution]) -> List[CandidateSolution]:
        """Produce the combination of solutions' strings.

        Args:
            lcs (List[CandidateSolution]): The list of solutions.

        Returns:
            List[CandidateSolution]: The list of solutions.
        """
        # Cartesian product of all strings, return merged string
        return [merge_solutions(lcs=x, modules_names=[m.name for m in self.modules]) for x in list(itertools.product(*lcs))]

    def _add_ll_strings(self,
                        cs: CandidateSolution) -> CandidateSolution:
        """Add the low-level string to a solution.

        Args:
            cs (CandidateSolution): The solution.

        Returns:
            CandidateSolution: The solution with the low-level string set.
        """
        ml_string = self.hl_solver.translator.transform(string=cs.string)
        cs.ll_string = self.ll_solver.solve(string=ml_string,
                                            iterations=1,
                                            strings_per_iteration=1,
                                            check_sat=False)[0].string
        return cs

    def _set_structure(self,
                       cs: CandidateSolution,
                       make_graph: bool = False) -> CandidateSolution:
        """Set the structure of the solution.

        Args:
            cs (CandidateSolution): The solution
            make_graph (bool, optional): Whether to plot the structure. Defaults to False.

        Returns:
            CandidateSolution: The solution with the structure set
        """
        base_position, orientation_forward, orientation_up = Vec.v3i(0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
        structure = Structure(origin=base_position,
                              orientation_forward=orientation_forward,
                              orientation_up=orientation_up)
        structure = LLStructureMaker(atoms_alphabet=self.ll_solver.atoms_alphabet,
                                     position=base_position).fill_structure(structure=structure,
                                                                            string=cs.ll_string)

        cs.set_content(content=structure)
        if make_graph:
            structure.show(title=cs.string)
        return cs

    def apply_rules(self,
                    starting_strings: List[str],
                    iterations: List[int],
                    create_structures: bool = False,
                    make_graph: bool = False) -> List[CandidateSolution]:
        """Apply the expansion rules for the hierarchical L-system.

        Args:
            starting_strings (List[str]): The starting strings (one per module).
            iterations (List[int]): The number of iterations to expand for (one per module).
            create_structures (bool, optional): Whether to create structures for each solution. Defaults to False.
            make_graph (bool, optional): Whether to plot the structure. Defaults to False.

        Returns:
            List[CandidateSolution]: The list of solutions.
        """
        assert len(starting_strings) == len(self.modules), f'Assumed wrong number of modules: have {len(self.modules)}, passed {len(starting_strings)}.'
        assert len(iterations) == len(self.modules), f'Assumed wrong number of modules: have {len(self.modules)}, passed {len(iterations)}.'
        # create solutions for each module and combine them
        solutions = self._produce_solutions_combinations([self.process_module(module=module,
                                                                              starting_string=starting_string,
                                                                              iterations=n_iterations) for module, starting_string, n_iterations in zip(self.modules, starting_strings, iterations)])
        # set low-level strings
        solutions = list(map(lambda cs: self._add_ll_strings(cs=cs), solutions))
        # if enabled, create the structures
        if create_structures:
            solutions = list(map(lambda cs: self._set_structure(cs=cs, make_graph=make_graph), solutions))
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
