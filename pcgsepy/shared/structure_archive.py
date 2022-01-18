from typing import List

from ..structure import Structure

class StructureArchive:
    def __init__(self):
        self._archive = {}
    
    def get_ll(self,
               hl_axiom: str) -> str:
        assert hl_axiom in self._archive.keys(), f'Axiom not in archive ({hl_axiom}).'
        return self._archive[hl_axiom].get('low_level', '')
    
    def get_structure(self,
                      hl_axiom: str) -> Structure:
        assert hl_axiom in self._archive.keys(), f'Axiom not in archive ({hl_axiom}).'
        return self._archive[hl_axiom].get('structure' None)
    
    def insert(self,
               hl_axiom: str,
               ll_axiom: str,
               structure: Structure) -> None:
        assert hl_axiom not in self._archive.keys(), f'Axiom already in archive ({hl_axiom}).'
        self._archive[hl_axiom] = {
            'low_level': ll_axiom,
            'structure': structure
        }
    
    def remove(self,
               hl_axiom: str) -> None:
        assert hl_axiom in self._archive.keys(), f'Axiom not in archive ({hl_axiom}).'
        self._archive.pop(hl_axiom)
    
    def refresh(self,
                to_keep: List[str]) -> None:
        for e in to_keep:
            self.remove(hl_axiom=e)