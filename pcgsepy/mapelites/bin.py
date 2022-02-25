from typing import List, Tuple
import numpy as np
from ..lsystem.solution import CandidateSolution
from ..config import BIN_POP_SIZE


class MAPBin:

    def __init__(self, bin_idx: Tuple[int, int], bin_size: Tuple[float, float]):
        self._feasible = []
        self._infeasible = []
        self.bin_idx = bin_idx
        self.bin_size = bin_size

    def __str__(self) -> str:
        vmin = (self.bin_idx[0] * self.bin_size[0],
                self.bin_idx[1] * self.bin_size[1])
        vmax = (vmin[0] + self.bin_size[0], vmin[1] + self.bin_size[1])
        return f'Bin {vmin}-{vmax} w/ {len(self._feasible)}f and {len(self._infeasible)}i cs'

    def __repr__(self) -> str:
        return str(self)

    def _reduce_pop(self,
                    pop: List[CandidateSolution],
                    maximize: bool = True) -> List[CandidateSolution]:
        if len(pop) > BIN_POP_SIZE:
            pop.sort(key=lambda x: x.c_fitness, reverse=maximize)
            pop = pop[:BIN_POP_SIZE]
        return pop

    def insert_cs(self, cs: CandidateSolution):
        if cs.is_feasible:
            if cs not in self._feasible:
                self._feasible.append(cs)
                self._feasible = self._reduce_pop(self._feasible)
        else:
            if cs not in self._infeasible:
                self._infeasible.append(cs)
                self._infeasible = self._reduce_pop(self._infeasible,
                                                    maximize=False)

    def age(self,
            diff: int = -1):
        for pop in [self._feasible, self._infeasible]:
            for cs in pop:
                cs.age += diff

    def remove_old(self):
        to_rem_f = [x for x in self._feasible if x.age <= 0]
        for cs in to_rem_f:
            self._feasible.remove(cs)
        to_rem_i = [x for x in self._infeasible if x.age <= 0]
        for cs in to_rem_i:
            self._infeasible.remove(cs)

    def get_metric(self,
                   metric: str,
                   use_mean: bool = True,
                   population: str = 'feasible'):
        op = np.mean if use_mean else np.max
        pop = self._feasible if population == 'feasible' else self._infeasible
        if metric == 'fitness':
            return op([cs.c_fitness for cs in pop]) if len(pop) > 0 else 0.
        elif metric == 'age':
            return op([cs.age for cs in pop]) if len(pop) > 0 else 0.
        elif metric == 'size':
            return len(pop)
        else:
            raise NotImplementedError(f'Unrecognized metric {metric}')

    def get_elite(self,
                  population: str = 'feasible'):
        pop = self._feasible if population == 'feasible' else self._infeasible
        get_max = True if population == 'feasible' else False
        return sorted(pop, key=lambda x: x.c_fitness, reverse=get_max)[0]

    def toggle_module_mutability(self,
                                 module: str):
        for pop in [self._feasible, self._infeasible]:
            for cs in pop:
                cs.hls_mod[module]['mutable'] = not cs.hls_mod[module]['mutable']