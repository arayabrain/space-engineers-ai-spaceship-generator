import logging
from typing import Any, Dict, List

from .actions import *
from .constraints import ConstraintHandler, ConstraintTime, ConstraintLevel
from .parser import HLParser, HLtoMLTranslator, LParser, LLParser
from ..structure import *
from ..common.vecs import orientation_from_str


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
            self.translator = HLtoMLTranslator(alphabet=self.atoms_alphabet,
                                               tiles_dims=extra_args['tiles_dimensions'],
                                               tiles_block_offset=extra_args['tiles_block_offset'])
            self.ll_rules = extra_args['ll_rules']

    def _forward_expansion(self,
                           axiom: str,
                           n: int,
                           dc_check: bool = False) -> None:
        for i in range(n):
            axiom = self.parser.expand(axiom=axiom)
        if dc_check and len([c for c in self.constraints if c.when == ConstraintTime.DURING]) > 0:
            logging.getLogger('base-logger').debug(msg=f'Expanding axiom {axiom}')
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
                    if c.needs_ll and self.translator:
                        ml_axiom = self.translator.transform(axiom=axiom)
                        ml_axiom = LLParser(rules=self.ll_rules).expand(axiom=ml_axiom)
                        s = c.constraint(axiom=ml_axiom,
                                         extra_args=c.extra_args)
                    else:
                        s = c.constraint(axiom=axiom,
                                         extra_args=c.extra_args)
                    logging.getLogger('base-logger').debug(msg=f'\t{c}:\t{s}')
                    sat[l] &= s
        return sat

    def solve(self,
              axiom: str,
              iterations: int,
              axioms_per_iteration: int = 1) -> List[str]:
        all_axioms = [axiom[:]]
        # forward expansion + DURING constraints check
        for i in range(iterations):
            logging.getLogger('base-logger').info(f'Expansion n.{i+1}/{iterations}; current number of axioms: {len(all_axioms)}')
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
                logging.getLogger('base-logger').debug(f'Finalizing axiom {axiom}')
                if not self._check_constraints(axiom=axiom,
                                               when=ConstraintTime.END)[ConstraintLevel.HARD_CONSTRAINT]:
                    to_rem.append(axiom)
            # remaining axioms are SAT
            for a in to_rem:
                all_axioms.remove(a)

        return all_axioms

    def add_constraint(self,
                       c: ConstraintHandler) -> None:
        self.constraints.append(c)
