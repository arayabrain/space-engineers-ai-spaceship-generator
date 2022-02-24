import logging
from typing import Any, Dict, List, Optional

from .constraints import ConstraintHandler, ConstraintTime, ConstraintLevel
from .parser import HLParser, HLtoMLTranslator, LParser, LLParser
from .solution import CandidateSolution


class LSolver:

    def __init__(self,
                 parser: LParser,
                 atoms_alphabet: Dict[str, Any],
                 extra_args: Dict[str, Any]):
        self.parser = parser
        self.atoms_alphabet = atoms_alphabet
        self.constraints = []
        self.inner_loops_during = 5
        self.inner_loops_end = 5
        self.translator = None
        if isinstance(self.parser, HLParser):
            self.translator = HLtoMLTranslator(
                alphabet=self.atoms_alphabet,
                tiles_dims=extra_args['tiles_dimensions'],
                tiles_block_offset=extra_args['tiles_block_offset'])
            self.ll_rules = extra_args['ll_rules']

    def _forward_expansion(self,
                           cs: CandidateSolution,
                           n: int,
                           dc_check: bool = False) -> Optional[CandidateSolution]:
        for i in range(n):
            logging.getLogger('base-logger').debug(msg=f'Expanding string {cs.string}')
            cs.string = self.parser.expand(string=cs.string)
        if dc_check and len([c for c in self.constraints if c.when == ConstraintTime.DURING]) > 0:
            if not self._check_constraints(cs=cs,
                                           when=ConstraintTime.DURING)[ConstraintLevel.HARD_CONSTRAINT]:
                cs = None  # do not continue expansion if it breaks hard constraints during expansion
        return cs

    def _check_constraints(self,
                           cs: CandidateSolution,
                           when: ConstraintTime,
                           keep_track: bool = False) -> Dict[ConstraintLevel, bool]:
        sat = {
            ConstraintLevel.SOFT_CONSTRAINT:
                True if not keep_track else [True, 0],
            ConstraintLevel.HARD_CONSTRAINT:
                True if not keep_track else [True, 0],
        }
        for lev in sat.keys():
            for c in self.constraints:
                if c.when == when and c.level == lev:
                    if c.needs_ll and self.translator:
                        cs.ll_string = self.translator.transform(string=cs.string)
                        cs.ll_string = LLParser(rules=self.ll_rules).expand(string=cs.ll_string)
                    s = c.constraint(cs=cs,
                                        extra_args=c.extra_args)
                    logging.getLogger('base-logger').debug(msg=f'\t{c}:\t{s}')
                    if keep_track:
                        sat[lev][0] &= s
                        sat[lev][1] += (
                            1 if lev == ConstraintLevel.HARD_CONSTRAINT else
                            0.5) if not s else 0
                    else:
                        sat[lev] &= s
        return sat

    def solve(self,
              string: str,
              iterations: int,
              strings_per_iteration: int = 1,
              check_sat: bool = True) -> List[CandidateSolution]:
        all_solutions = [CandidateSolution(string=string)]
        # forward expansion + DURING constraints check
        for i in range(iterations):
            logging.getLogger('base-logger').info(f'Expansion n.{i+1}/{iterations}; current number of strings: {len(all_solutions)}')
            new_all_solutions = []
            for cs in all_solutions:
                for _ in range(strings_per_iteration):
                    new_cs = CandidateSolution(string=cs.string[:])
                    new_cs = self._forward_expansion(cs=new_cs,
                                                     n=1,
                                                     dc_check=i > 0 and check_sat)
                    if new_cs is not None:
                        new_all_solutions.append(new_cs)
            all_solutions = new_all_solutions
            all_solutions = list(set(all_solutions))  # remove duplicates

        # END constraints check + possible backtracking
        if check_sat and len([c for c in self.constraints if c.when == ConstraintTime.END]) > 0:
            to_rem = []
            for cs in all_solutions:
                logging.getLogger('base-logger').debug(f'Finalizing string {cs.string}')
                if not self._check_constraints(cs=cs,
                                               when=ConstraintTime.END)[ConstraintLevel.HARD_CONSTRAINT]:
                    to_rem.append(cs)
            # remaining strings are SAT
            for a in to_rem:
                all_solutions.remove(a)

        return all_solutions

    def set_constraints(self,
                        cs: List[ConstraintHandler]) -> None:
        self.constraints = cs