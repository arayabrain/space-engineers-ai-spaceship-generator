from typing import Any, Dict, Optional

import numpy as np

from ..structure import Structure


class CandidateSolution:

    def __init__(self,
                 string: str,
                 content: Optional[Structure] = None):
        self.string = string
        self._content = content
        self.representation = []
        self.fitness = []
        self.c_fitness = 0.
        self.b_descs = (0., 0.)
        self.is_feasible = True
        self.age = 0
        self.ll_string = ''
        self.hls_mod = {}  # keys: 'string', 'mutable'
        self.ncv = 0  # number of constraints violated
        self.parents = []
        self.n_offspring = 0
        self.n_feas_offspring = 0
        
        self.size = ()
        self.n_blocks = 0

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return str(self) + f'; fitness: {self.c_fitness}; is_feasible: {self.is_feasible}'

    def __eq__(self,
               other: 'CandidateSolution') -> bool:
        if isinstance(other, CandidateSolution):
            return self.string == other.string
        return False

    def __hash__(self):
        return hash(self.string)

    def set_content(self,
                    content: Structure):
        if self._content:
            raise Exception('Structure already exists for this CandidateSolution.')
        else:
            self._content = content
            self.size = self._content._max_dims
            self.n_blocks = len(self._content._blocks)

    @property
    def content(self) -> Structure:
        if self._content:
            return self._content
        else:
            raise NotImplementedError('Structure has not been set yet.')
    
    def to_json(self) -> Dict[str, Any]:
        return {
            'string': self.string,
            'representation': self.representation,
            'fitness': self.fitness,
            'c_fitness': self.c_fitness,
            'b_descs': self.b_descs,
            'is_feasible': self.is_feasible,
            'age': self.age,
            'll_string': self.ll_string,
            'hls_mod': self.hls_mod,
            'ncv': self.ncv,
            'parents': [p.to_json() for p in self.parents],
            'n_offsprings': self.n_offspring,
            'n_feas_offsprings': self.n_feas_offspring
        }
    
    @staticmethod
    def from_json(my_args: Dict[str, Any]) -> 'CandidateSolution':
        cs = CandidateSolution(string=my_args['string'],
                               content=None)
        cs.representation = my_args['representation']
        cs.fitness = my_args['fitness']
        cs.c_fitness = my_args['c_fitness']
        cs.b_descs = my_args['b_descs']
        cs.is_feasible = my_args['is_feasible']
        cs.age = my_args['age']
        cs.ll_string = my_args['ll_string']
        cs.hls_mod = my_args['hls_mod']
        cs.ncv = my_args['ncv']
        cs.parents = [CandidateSolution.from_json(args=p) for p in my_args['parents']]
        cs.n_offsprings = my_args['n_offsprings']
        cs.n_feas_offsprings = my_args['n_feas_offsprings']
        return cs