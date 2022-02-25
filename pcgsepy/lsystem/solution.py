from typing import Optional

from ..structure import Structure


class CandidateSolution:

    def __init__(self,
                 string: str,
                 content: Optional[Structure] = None):
        self.string = string
        self._content = content

        self.c_fitness = 0.
        self.b_descs = (0., 0.)
        self.is_feasible = True
        self.age = 0
        
        self.ll_string = ''
        self.hls_mod = {}  # keys: 'string', 'mutable'
        self.ncv = 0  # number of constraints violated

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return str(self)[:50] + f'; fitness: {self.c_fitness}; is_feasible: {self.is_feasible}'

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

    @property
    def content(self) -> Structure:
        if self._content:
            return self._content
        else:
            raise NotImplementedError('Structure has not been set yet.')