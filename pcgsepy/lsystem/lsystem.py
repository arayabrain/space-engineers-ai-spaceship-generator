from typing import Dict, Tuple

from .rules import RuleMaker
from .parser import LParser
from .solver import LSolver
from .constraints import ConstraintHandler
from .structure_maker import StructureMaker
from ..structure import Structure
from ..common.vecs import Orientation, Vec


class LSystem:
    def __init__(self,
                 hl_rules: str,
                 ll_rules: str,
                 alphabet: Dict[str, any]):
        self.hl_rules = RuleMaker(ruleset=hl_rules).get_rules()
        self.ll_rules = RuleMaker(ruleset=ll_rules).get_rules()
        self.hl_parser = LParser(rules=self.hl_rules)
        self.ll_parser = LParser(rules=self.ll_rules)
        self.hlsolver = LSolver(atoms_alphabet=alphabet,
                                parser=self.hl_parser)
        self.llsolver = LSolver(atoms_alphabet=alphabet,
                                parser=self.ll_parser)
    
    def add_hl_constraint(self,
                          c: ConstraintHandler) -> None:
        self.hlsolver.add_constraint(c)
    
    def add_ll_constraint(self,
                          c: ConstraintHandler) -> None:
        self.llsolver.add_constraint(c)
    
    def apply_rules(self,
                    starting_axiom: str,
                    iterations: int = 1,
                    make_graph: bool = False) -> Structure:        
        base_position, orientation_forward, orientation_up = Vec.v3i(0, 0, 0), Orientation.FORWARD.value, Orientation.UP.value
        structure = Structure(origin=base_position,
                              orientation_forward=orientation_forward,
                              orientation_up=orientation_up)
        print('### HL SOLVER ###')
        hl_axiom = self.hlsolver.solve(axiom=starting_axiom,
                                       iterations=iterations,
                                       axioms_per_iteration=2)
        print('\n### LL SOLVER ###')
        ll_axiom = self.llsolver.solve(axiom=hl_axiom,
                                       iterations=1)   
                       
        structure = StructureMaker(atoms_alphabet=self.llsolver.atoms_alphabet,
                                   position=base_position).fill_structure(structure=structure,
                                                                          axiom=ll_axiom)
        
        structure.sanify()
        
        if make_graph:
            structure.show(title=hl_axiom)
        
        return structure