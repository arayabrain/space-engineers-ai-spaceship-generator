import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from pcgsepy.lsystem.rules import StochasticRules

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
        logging.getLogger('solver').debug(f'[{__name__}._forward_expansion] Expanding {cs.string=}.')
        for i in range(n):
            cs.string = self.parser.expand(string=cs.string)
        if dc_check and len([c for c in self.constraints if c.when == ConstraintTime.DURING]) > 0:
            if not self._check_constraints(cs=cs,
                                           when=ConstraintTime.DURING)[ConstraintLevel.HARD_CONSTRAINT][0]:
                cs = None  # do not continue expansion if it breaks hard constraints during expansion
        return cs

    def _check_constraints(self,
                           cs: CandidateSolution,
                           when: ConstraintTime,
                           keep_track: bool = False) -> Dict[ConstraintLevel, List[Union[bool, int]]]:
        logging.getLogger('solver').debug(f'[{__name__}._check_constraints] Checking constraints on {cs.string=}.')
        sat = {
            ConstraintLevel.SOFT_CONSTRAINT: [True, 0],
            ConstraintLevel.HARD_CONSTRAINT: [True, 0],
        }
        for lev in sat.keys():
            for c in self.constraints:
                if c.when == when and c.level == lev:
                    s = c.constraint(cs=cs,
                                     extra_args=c.extra_args)
                    logging.getLogger('solver').debug(f'[{__name__}._forward_expansion] \t{c}:\t{s}.')
                    sat[lev][0] &= s
                    if keep_track:
                        sat[lev][1] += (
                            1 if lev == ConstraintLevel.HARD_CONSTRAINT else
                            0.5) if not s else 0
        return sat

    def solve(self,
              string: str,
              iterations: int,
              strings_per_iteration: int = 1,
              check_sat: bool = True) -> List[CandidateSolution]:
        all_solutions = [CandidateSolution(string=string)]
        # forward expansion + DURING constraints check
        for i in range(iterations):
            logging.getLogger('solver').debug(f'[{__name__}.solve] Expansion {i+1}/{iterations}; current number of strings: {len(all_solutions)}')
            new_all_solutions = []
            for cs in all_solutions:
                for _ in range(strings_per_iteration):
                    new_cs = CandidateSolution(string=cs.string[:])
                    new_cs = self._forward_expansion(cs=new_cs,
                                                     n=1,
                                                     dc_check=check_sat and i > 0)
                    if new_cs is not None:
                        new_all_solutions.append(new_cs)
            all_solutions = new_all_solutions
            all_solutions = list(set(all_solutions))  # remove duplicates

        # END constraints check + possible backtracking
        if check_sat and len([c for c in self.constraints if c.when == ConstraintTime.END]) > 0:
            to_keep = np.zeros(shape=len(all_solutions), dtype=np.bool8)
            for i, cs in enumerate(all_solutions):
                logging.getLogger('solver').debug(f'[{__name__}.solve] Finalizing string {cs.string}')
                to_keep[i] = self._check_constraints(cs=cs,
                                                     when=ConstraintTime.END)[ConstraintLevel.HARD_CONSTRAINT][0]
            # remaining strings are SAT
            all_solutions = [cs for i, cs in enumerate(all_solutions) if to_keep[i]]
            
        return all_solutions

    def set_constraints(self,
                        cs: List[ConstraintHandler]) -> None:
        self.constraints = cs
    
    def to_json(self) -> Dict[str, Any]:
        j = {
            'has_hl_parser': isinstance(self.parser, HLParser),
            'rules': self.parser.rules.to_json(),
            'atoms_alphabet': self.atoms_alphabet
        }
        if isinstance(self.parser, HLParser):
            j['extra_args'] = {
                'tiles_dimensions': self.translator.td,
                'tiles_block_offset': self.translator.tbo,
                'll_rules': self.ll_rules
            }
        return j
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'LSolver':
        parser = HLParser if my_args['has_hl_parser'] else LLParser
        return LSolver(parser=parser(rules=StochasticRules.from_json(my_args['rules'])),
                       atoms_alphabet=my_args['atoms_alphabet'],
                       extra_args=my_args.get('extra_args', {}))